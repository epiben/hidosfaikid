import numpy as np
import json
import optuna
import pandas as pd
import pickle

from concurrent.futures import ProcessPoolExecutor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from time import strftime, localtime, time, mktime
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sqlalchemy import create_engine

from utils import train_model, StratifiedGroupKFold, status

# ===

def optuna_status(trial_number, cv_fold, log_file=snakemake.log[0]):
	
	with open(log_file, 'r') as f: 
		log = [x.strip().split(' ') for x in f.read().splitlines()]
	    
	n_rows = len(log) # to reconstruct the matrix later
	log = [[char for char in x] for y in log for x in y]
	log[trial_number][cv_fold] = '='
	log = [''.join(x) for x in log]
	
	n_cols = len(log) // n_rows
	sep = [x for y in [[' '] * (n_cols-1) + ['\n']] * n_rows for x in y]
	log = ''.join(x + y for x,y in zip(log, sep))
	
	print(log)
	with open(log_file, 'w') as f: f.write(log)

def optuna_objective(trial, snakemake=None, study_name=None, datasets=dict(), db_conn=None):
	
	# Hyperparameters
	hp = dict()
	hp["optimiser_name"] = trial.suggest_categorical("optimiser_name", ["Adam", "RMSprop"])
	hp["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-6, 0.1)
	hp["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
	hp["l2_penalty"] = trial.suggest_loguniform("l2_penalty", 1e-6, 1e-2)
	hp["class_handling"] = trial.suggest_categorical("class_handling", ["smote", "random_over", "class_weighting", "near_miss", "random_under", "none"])
	
	## Only applicable to MLP models
	if snakemake.wildcards["model_type"] != "linear":
		hp["activation"] = trial.suggest_categorical("activation", ["sigmoid", "tanh"])
		hp["n_hidden_layers"] = trial.suggest_int("n_hidden_layers", 1, 4)
		hp["n_hidden_nodes"] = trial.suggest_categorical("n_hidden_nodes", [8, 16, 32, 64, 128, 256])
		hp["network_shape"] = trial.suggest_categorical("network_shape", ["rectangular"]) #trial.suggest_categorical("network_shape", ["rectangular", "triangular"])
		
	# Set up cross-validation
	datasets["dev"] = datasets["dev"].sample(frac=1, axis=1, random_state=42).reset_index(drop=True)
		# shuffle here because GroupKFold doesn't allow for that
	y_dev = datasets["dev"][outcome_variable].values
	optim_metric = list() # will return the mean of this to Optuna (different for Poisson and binary regression)
	cv_fold = 0
	
	# kfcv = StratifiedKFold(n_splits=snakemake.params["n_cv_folds"], random_state=42, shuffle=True)
	kfcv = StratifiedGroupKFold(n_splits=snakemake.params["n_cv_folds"])
	for train_idx, val_idx in kfcv.split(np.zeros(len(y_dev)), y_dev, groups=datasets["dev"].person_id):
		cv_fold += 1
		# status(f"Trial number: {trial.number} \tCV fold: {cv_fold}", snakemake.log[0])
		optuna_status(trial.number, cv_fold)
		
		# Define callbacks (weights_path is specific to each CV fold so must be done inside loop)
		weights_dir = snakemake.params["weights_dir"]
		weights_fpath = f"{weights_dir}/{study_name}__trial_{trial.number}__cv_fold_{cv_fold}__best_weights.hdf5"
		checkpoint = ModelCheckpoint(
			monitor="val_loss",
			filepath=weights_fpath,
			save_best_only=True,
			save_weights_only=False,
			mode="min"
		)
		early_stopping = EarlyStopping(
			monitor="val_loss", 
			min_delta=0, 
			patience=snakemake.params["early_stop_patience"], 
			mode="min", 
			restore_best_weights=False
		)
		
		# Assign train and val subsets to the datasets dict
		datasets["train"] = datasets["dev"].iloc[train_idx]
		datasets["val"] = datasets["dev"].iloc[val_idx]
		
		hist = train_model(
			datasets=datasets, 
			hp=hp, # hyperparameter dict
			outcome_type=snakemake.wildcards["outcome_type"], # zip, poisson, binary 
			model_type = snakemake.wildcards["model_type"], # linear, mlp
			outcome_variable = snakemake.wildcards["outcome_variable"], # r, daily_rate_*
			weights_fpath=weights_fpath,
			n_epochs=snakemake.params["n_epochs"],
			cv_fold=cv_fold,
			callbacks=[checkpoint, early_stopping],
			study_name=study_name,
			trial_number=trial.number,
			db_conn=db_conn
		)
		optim_metric.append(np.min(hist.history["val_loss"])) 
			
	# Return mean of the metric across CV folds to Optuna for optimisation
	return np.mean(optim_metric) 

DBNAME = snakemake.params["dbname"]
DBUSER = snakemake.params["dbuser"]
DBSCHEMA = snakemake.params["dbschema"]

psql_url = f"postgresql://{DBUSER}@dbserver/{DBNAME}?options=-c%20search_path={DBSCHEMA}"
engine = create_engine(psql_url)

if __name__ == "__main__":
	
	outcome_type = snakemake.wildcards["outcome_type"]
	model_type = snakemake.wildcards["model_type"]
	outcome_variable = snakemake.wildcards["outcome_variable"]
	study_name = f"{outcome_type}__{model_type}__{outcome_variable}"
	
	storage = optuna.storages.RDBStorage(url=psql_url, engine_kwargs={"pool_size": 0})
	
	try:
		optuna.delete_study(storage=storage, study_name=study_name)
	except:
		pass

	study = optuna.create_study(
		storage=storage, study_name=study_name, direction="minimize",
		pruner=optuna.pruners.HyperbandPruner(min_resource=15, reduction_factor=3),
		sampler=optuna.samplers.TPESampler(multivariate=True, seed=42)
	)
	study.set_user_attr("contributors", ["benkaa"])

	n_jobs = snakemake.threads
	n_trials = snakemake.params.n_optuna_trials // n_jobs

	def optimize(n_trials):
		
		study = optuna.load_study(storage=psql_url, study_name=study_name)
		
		with engine.connect() as connection:
			datasets = {x: pd.read_sql("data_" + x, connection) for x in ("dev", "test", "test_new")}
			study.optimize(
				lambda trial: optuna_objective(trial, snakemake, study_name, datasets, connection), 
				n_trials=n_trials, 
				n_jobs=1
			)

	# Setup log file with "matrix" of trials and CV folds
	with open(snakemake.log[0], "a") as f:
		n_cols = min(10, snakemake.params["n_optuna_trials"])
		n_rows = snakemake.params["n_optuna_trials"] // n_cols
		n_cv_folds = snakemake.params["n_cv_folds"]
		f.write((("[" + "-" * n_cv_folds + "] ") * n_cols + "\n") * n_rows)
    	
	with ProcessPoolExecutor(max_workers=n_jobs) as executor:
		executor.map(optimize, [n_trials] * n_jobs)

	is_pruned = lambda t: t.state == optuna.trial.TrialState.PRUNED
	is_complete = lambda t: t.state == optuna.trial.TrialState.COMPLETE
	pruned_trials = [t for t in study.trials if is_pruned(t)]
	complete_trials = [t for t in study.trials if is_complete(t)]
	
	status("Study statistics: ", snakemake.log[0])
	status(f"  Number of finished trials: {len(study.trials)}", snakemake.log[0])
	status(f"  Number of pruned trials: {len(pruned_trials)}", snakemake.log[0])
	status(f"  Number of complete trials: {len(complete_trials)}", snakemake.log[0])

	# Various outputs
	status("Creating final output and saving it", snakemake.log[0])
	trials_df = study.trials_dataframe(multi_index=False)
	trials_df = pd.concat([pd.DataFrame({"study_name": [study_name]*trials_df.shape[0]}), trials_df], axis=1)
	trials_df.to_csv(snakemake.output["fpath_trials_df"], sep="\t", index=True)
	
	with engine.connect() as connection:
		pd.DataFrame({
			"study_name": study_name,
			"trial_id": study.best_trial._trial_id,
			"hyperparameters": json.dumps(study.best_trial.params)
		}, index=[0]).to_sql("best_trials", connection, if_exists="append", index=False)
	
	with open(snakemake.output["fpath_pickle"], "wb") as f: 
		pickle.dump(study, f, protocol=pickle.HIGHEST_PROTOCOL)
	
	status("Done", snakemake.log[0])	
	
		
