import json
import numpy as np
import pandas as pd
import shap 
import sys

from sqlalchemy import create_engine

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, precision_score, recall_score

from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.layers import Activation, Lambda, ReLU
from tensorflow.keras.models import Model, load_model

from utils import pick_features, handle_imbalance, remove_old_results_from_db

class early_stopping_by_metric(Callback):
	"""
	A helper callback allowing us to stop training only when the defined metric reaches a certain value.
	"""
	def __init__(self, monitor="loss", value=0.00001):
		super(Callback, self).__init__()
		self.monitor, self.value = monitor, value
		
	def on_epoch_end(self, epoch, logs={}):
		if logs.get(self.monitor) <= self.value:
			self.model.stop_training = True

# sys.stdout = open(snakemake.log[0], "w")

DBNAME = snakemake.params["dbname"]
DBUSER = snakemake.params["dbuser"]
DBSCHEMA = snakemake.params["dbschema"]
psql_url = f"postgresql://{DBUSER}@dbserver/{DBNAME}?options=-c%20search_path={DBSCHEMA}"
engine = create_engine(psql_url) 

with engine.connect() as connection:
	outcome_type = snakemake.wildcards["outcome_type"]
	model_type = snakemake.wildcards["model_type"]
	outcome_variable = snakemake.wildcards["outcome_variable"]
	optuna_study_name = f"{outcome_type}__{model_type}__{outcome_variable}"
	
	hp = pd.read_sql(f"SELECT hyperparameters FROM best_trials WHERE study_name = '{optuna_study_name}'", connection)
	hp = json.loads(hp.hyperparameters[0])
	
	datasets = {x: pd.read_sql("data_" + x, connection) for x in ("dev", "test", "test_new")}
	features = {k: pick_features(v).values for k,v in datasets.items()}
	y = {k: v[outcome_variable].values for k,v in datasets.items()}
	
	features["dev"], y["dev"], class_weights = \
		handle_imbalance(features["dev"], y["dev"], hp["class_handling"])

	# Read settings needed to pick up where best CV fold stopped so we can load the model	
	cv_model_to_continue_training = pd.read_sql(f"""
		WITH cte_loss AS (
			SELECT 
				bt.study_name
				, trial_number
				, cv_fold
				, (eval_val::json->>'loss')::float AS target_loss
			FROM best_trials AS bt
			INNER JOIN trials
				ON trials.trial_id = bt.trial_id
			INNER JOIN training_summaries AS ts
				ON ts.study_name = bt.study_name
				AND ts.trial_number = trials.number
		) 
		SELECT *
		FROM (
			SELECT *, ROW_NUMBER() OVER(PARTITION BY study_name ORDER BY target_loss) AS rn 
			FROM cte_loss
		) AS loss
		WHERE rn = 1
			AND study_name = '{optuna_study_name}';
	""", connection)
	cv_fold = int(cv_model_to_continue_training.cv_fold)
	trial_number = int(cv_model_to_continue_training.trial_number)
	target_loss = float(cv_model_to_continue_training.target_loss)
	model_crude = load_model(snakemake.params["weights_dir"] + \
		f"/{optuna_study_name}__trial_{trial_number}__cv_fold_{cv_fold}__best_weights.hdf5")
	
	# Define callbacks
	study_name = f"{outcome_type}__{model_type}__{outcome_variable}__final"
	weights_fpath = snakemake.params["weights_dir"] + f"/final__{study_name}__best_weights.hdf5"
	checkpoint = ModelCheckpoint( # needed to save the model so we can re-invoke it below for Platt calibration
		monitor="loss",
		filepath=weights_fpath,
		save_best_only=True,
		save_weights_only=False,
		mode="min"
	)
	early_stopping = early_stopping_by_metric(value=target_loss)
	
	# Continue training
	hist = model_crude.fit(
		x=features["dev"], y=y["dev"], 
		verbose=False,
		epochs=500, 
		batch_size=hp["batch_size"], 
		validation_data=None,
		callbacks=[checkpoint, early_stopping],
		class_weight=class_weights
	)
	
	# Platt recalibration
	# acoeff, bcoeff = _sigmoid_calibration(model_crude.predict(features["dev"]), y["dev"])
	# 
	# platt = model_crude.layers[-1].output
	# platt = Lambda(lambda x: -(acoeff*x + bcoeff))(platt)
	# platt = Activation("sigmoid")(platt)
	# model_platt = Model(inputs=model_crude.layers[0].input, outputs=platt)
	# 
	# # Isotonic regression
	# iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
	# iso_reg = iso_reg.fit(model_crude.predict(features["dev"]).squeeze(), y["dev"])
	
	# Make predictions and save in database
	def make_pred_df (y_true, features, dataset, mod, pred_type=None):
		y_pred = pd.DataFrame(mod.predict(features), columns=["y_pred"])
		# if pred_type == "isotonic":
		# 	y_pred = pd.DataFrame(iso_reg.predict(y_pred.squeeze()), columns=["y_pred"])
		context = pd.DataFrame({
			"study_name": study_name, 
			"dataset": dataset, 
			"admission_id": datasets[dataset].admission_id,
			"pred_type": pred_type, 
			"y_true": y_true})
		return pd.concat([context, y_pred], axis="columns")
	
	preds = pd.concat([make_pred_df(y[k], features[k], k, model_crude, "crude") for k in datasets.keys()])
	remove_old_results_from_db("predictions", "study_name", study_name, connection)
	preds.to_sql("predictions", connection, if_exists="append", index=False)
	
	# Evaluation metrics
	def get_metrics(k, mod=model_crude, pred_type="crude"):
		# First, find best threshold as per Matthew's correlation coefficient
		y_pred_train = mod.predict(features["dev"])
		y_true_train = y["dev"]
		best_mcc = 0
		threshold = 0
		for th in np.linspace(0, 1, 100, endpoint=False):
			y_binary = (y_pred_train > th).astype("int32")
			mcc = matthews_corrcoef(y_true_train, y_binary)
			if mcc > best_mcc:
				threshold = th

		# Then, use threshold to compute metrics
		y_true = y[k]
		y_pred = mod.predict(features[k])
		y_pred_binary = (y_pred > threshold).astype("int32")
		return pd.DataFrame({
			"study_name": study_name,
			"dataset": k, 
			"pred_type": pred_type,
			"auroc": roc_auc_score(y_true, y_pred),
			"avg_prec_score": average_precision_score(y_true, y_pred),
			"mcc": matthews_corrcoef(y_true, y_pred_binary),
			"precision": precision_score(y_true, y_pred_binary),
			"recall": recall_score(y_true, y_pred_binary),
			"threshold": threshold,
			"n_epochs": len(hist.history["loss"])
		}, index=[0])
	metrics = pd.concat([get_metrics(k, model_crude, "crude") for k in datasets.keys()])
	remove_old_results_from_db("evaluation_metrics", "study_name", study_name, connection)
	metrics.to_sql("evaluation_metrics", connection, if_exists="append", index=False)
	
	# Make SHAP explanations
	features = {k: pick_features(v) for k,v in datasets.items()} # needed again without converting to np array
	try:
		n_samples = min(snakemake.params["n_shap_background"], features["dev"].shape[0])
		background_set = features["dev"].sample(n_samples, random_state=42)
	except:
		background_set = features["dev"]
	
	try:
		n_explanations = snakemake.params["n_explanations"]
	except:
		n_explanations = features["test"].shape[0]
	
	print("Creating explainer")
	e = shap.DeepExplainer(model_crude, background_set.values)
	
	print("Computing shap values")
	shap_values = e.shap_values(features["test"][:n_explanations].values)[0]
	
	print("Creating combined dataframe and writing to database")
	context = pd.DataFrame({
		"study_name": study_name,
		"dataset": "test",
		"pred_type": "crude",
		"admission_id": datasets["test"].admission_id[:n_explanations],
		"y_true": y["test"][:n_explanations],
		"y_pred": model_crude.predict(features["test"][:n_explanations]).squeeze(),
		"base_value": float(e.expected_value)
	})
	shap_df = pd.concat([context, pd.DataFrame(shap_values, columns=background_set.columns)], axis=1)
	remove_old_results_from_db("shap_values", "study_name", study_name, connection)
	shap_df.to_sql("shap_values", connection, if_exists="append", index=False)
	
	# Write training history as output monitored by snakemake
	pd.DataFrame(hist.history).to_csv(snakemake.output[0], sep="\t", index=False)

# sys.stdout.close()
