import copy
import json
import numpy as np
import os
import pandas as pd
import tensorflow as tf
# import tensorflow_probability as tfp

from random import randint

from math import log
from scipy.optimize import fmin_bfgs
from scipy.special import expit
from scipy.special import xlogy

from sklearn.utils import class_weight
from sklearn.utils.validation import column_or_1d

from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Input, Dropout, Multiply, Concatenate, Activation, ReLU
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import L1L2, L1, L2

from time import strftime, localtime, time, mktime

# For some reason it works to try to import imblearn and in the except just import the functions we need (wtf?)
try: 
	import imblearn
except: 
	from imblearn.over_sampling import SMOTE, RandomOverSampler
	from imblearn.under_sampling import NearMiss, RandomUnderSampler

# tfd = tfp.distributions

# ========== FUNCTIONS =========

def status(out, log_file):
	now = strftime("%Y-%m-%d_%H:%M:%S", localtime(time()))
	try:
		with open(log_file, 'a') as f:
			f.write(f"{now}\t{out}\n")
	except:
		pass
	print(f"{now}\t{out}")

def give_correct_dtype(v): 
	try:
		v = float(v)
	except ValueError:
		return(v)
	if int(v) == v:
		return(int(v))
	return(v)
	
def pick_features(d):
	"""
	Drop columns from input Pandas data frame that should not be used as features, as return as Numpy array.
	"""
	cols_to_drop = ["admission_id", "person_id", "r", "time_at_risk", "daily_rate_not_zero"] + \
		[f"daily_rate_geq_{x}" for x in (1, 2, 3, 5)]
	return d.drop(cols_to_drop, axis=1)

def handle_imbalance(features, y, mechanism):
	class_weights = None
	if mechanism == "class_weighting":
		class_weights = dict(enumerate(
			class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)
		))
		
	try:
		if mechanism == "smote":
			sampler = SMOTE()
		elif mechanisms == "random_over":
			sampler = RandomOverSampler(random_state=42)
		elif mechanism == "near_miss":
			sampler = NearMiss(version=1, n_neighbors=5)
		elif mechanism == "random_under":
			sampler = RandomUnderSampler(random_state=42)
		return sampler.fit_resample(features, y), class_weigths
	except:
		return features, y, class_weights

def remove_old_results_from_db(table_name, col_name, study_name, conn):
	try: # the table might not exist yet
		conn.execute(f"BEGIN; DELETE FROM {table_name} WHERE {col_name} = '{study_name}'; COMMIT;")
		print(f"Old results removed from table {table_name}.")
	except:
		print(f"Table {table_name} doesn't exist; nothing done dbserver-side.")
		pass
	
def train_model(datasets=None, 
				hp=None, # hyperparameter dict
				outcome_type=None,
				model_type=None,
				outcome_variable=None,
				weights_fpath=None,
				n_epochs=None,
				cv_fold=None,
				callbacks=None,
				study_name=None,
				trial_number=None,
				db_conn=None):
	"""
	
	"""
	
	# Wrange data as required
	datasets = copy.deepcopy(datasets) # don't destroy original dict
	datasets.pop("dev", None)
	
	features = {k: pick_features(v).values for k,v in datasets.items()}
	y = {k: v[outcome_variable].values for k,v in datasets.items()}

	# Model architecture
	inputs_data = Input(shape=(features["train"].shape[1], ), name="features")

	if model_type == "linear":
		model_outputs = Dense(
			1, 
			activation="sigmoid",
			kernel_regularizer=L2(hp["l2_penalty"]), 
			kernel_initializer=RandomNormal(seed=42)
		)(inputs_data)
	else:
		if hp["activation"] == "relu6":
			activation = ReLU(max_value=6)
		else:
			activation = hp["activation"]
			
		n_nodes_reduction_factor = {"rectangular": 1, "triangular": 2}
		
		for i in range(hp["n_hidden_layers"]):
			if i == 0:
				n_nodes_this_layer = hp["n_hidden_nodes"]
				hidden = Dense(
					n_nodes_this_layer, 
					activation=activation,
					kernel_regularizer=L2(hp["l2_penalty"]), 
					kernel_initializer=RandomNormal(seed=42)
				)(inputs_data)
			else:
				n_nodes_this_layer /= n_nodes_reduction_factor[hp["network_shape"]]
				hidden = Dense(
					n_nodes_this_layer, 
					activation=activation,
					kernel_regularizer=L2(hp["l2_penalty"]), 
					kernel_initializer=RandomNormal(seed=42)
				)(hidden)
		model_outputs = Dense(
			1, 
			activation="sigmoid", 
			kernel_regularizer=L2(hp["l2_penalty"]),
			kernel_initializer=RandomNormal(seed=42)
		)(hidden)
	
	# Handling class imbalances
	features["train"], y["train"], class_weights = \
		handle_imbalance(features["train"], y["train"], hp["class_handling"])
	
	# Construct and train model
	if "val" in datasets:
		validation_data = (features["val"], y["val"])
	else:
		validation_data = None
		
	model = Model(inputs=inputs_data, outputs=model_outputs)
	model.compile(optimizer=getattr(optimizers, hp["optimiser_name"])(learning_rate=hp["learning_rate"]), 
				  loss="binary_crossentropy",
				  metrics=["accuracy", AUC(curve="ROC", name="auroc"), AUC(curve="PR", name="auprc")])
	model.save_weights(weights_fpath) # ensure we have some best weights (initial ones will always be non-NaN)
	
	model.summary() # FIX: consider save model graph as pdf at this point
	hist = model.fit(
		x=features["train"], y=y["train"], 
		verbose=False,
		epochs=n_epochs, 
		batch_size=hp["batch_size"], 
		validation_data=validation_data,
		callbacks=callbacks,
		class_weight=class_weights
	)
	model.load_weights(weights_fpath)
	
	# Compute losses and metrics and save in database
	try:
		eval_val = json.dumps(model.evaluate(features["val"], y["val"], verbose=False, return_dict=True))
	except:
		eval_val = None
		
	try:
		effective_n_epochs = len(hist.history["loss"]) - (snakemake.params["early_stop_patience"] - 1)
	except:
		effective_n_epochs = len(hist.history["loss"])
	
	training_summary = pd.DataFrame({
		"study_name": study_name,
		"hyperparameters": json.dumps(hp),
		"eval_train": json.dumps(model.evaluate(features["train"], y["train"], verbose=False, return_dict=True)),
		"eval_val": eval_val,
		"eval_test": json.dumps(model.evaluate(features["test"], y["test"], verbose=False, return_dict=True)),
		"eval_test_new": json.dumps(model.evaluate(features["test_new"], y["test_new"], verbose=False, return_dict=True)),
		"history": json.dumps(hist.history),
		"n_epochs": effective_n_epochs,
		"trial_number": trial_number,
		"cv_fold": cv_fold,
	}, index=[0])
	
	tries = 1
	while tries <= 5:
		try:
			# This happens when two parallel processes to create the table concurrently
			training_summary.to_sql("training_summaries", db_conn, if_exists="append", index=False)
		except:
			if tries < 5:
				print("Couldn't save training summary to database, will retry in 5 seconds.")
				time.sleep(5)
			else:
				raise Exception("Couldn't save training summary to database after 5 tries.")
		finally:
			tries += 1
	        
	return(hist)
	
# This implementation of StratifiedGroupKFold hasn't yet made it into sklearn;
# code from https://github.com/scikit-learn/scikit-learn/issues/13621#issuecomment-656094573
from collections import Counter, defaultdict

import numpy as np

from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits
from sklearn.utils.validation import check_random_state


class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds iterator variant with non-overlapping groups.

    This cross-validation object is a variation of StratifiedKFold that returns
    stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.

    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).

    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = StratifiedGroupKFold(n_splits=3)
    >>> for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [2 2 4 5 5 5 5 6 6 7]
           [1 1 1 0 0 0 0 0 0 0]
     TEST: [1 1 3 3 3 8 8]
           [0 0 1 1 1 0 0]
    TRAIN: [1 1 3 3 3 4 5 5 5 5 8 8]
           [0 0 1 1 1 1 0 0 0 0 0 0]
     TEST: [2 2 6 6 7]
           [1 1 0 0 0]
    TRAIN: [1 1 2 2 3 3 3 6 6 7 8 8]
           [0 0 1 1 1 1 1 0 0 0 0 0]
     TEST: [4 5 5 5 5]
           [1 0 0 0 0]

    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).

    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    # Implementation based on this kaggle kernel:
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def _iter_test_indices(self, X, y, groups):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, group in zip(y, groups):
            y_counts_per_group[group][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        rng = check_random_state(self.random_state)
        if self.shuffle:
            rng.shuffle(groups_and_y_counts)

        for group, y_counts in sorted(groups_and_y_counts,
                                      key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                y_counts_per_fold[i] += y_counts
                std_per_label = []
                for label in range(labels_num):
                    std_per_label.append(np.std(
                        [y_counts_per_fold[j][label] / y_distr[label]
                         for j in range(self.n_splits)]))
                y_counts_per_fold[i] -= y_counts
                fold_eval = np.mean(std_per_label)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(group)

        for i in range(self.n_splits):
            test_indices = [idx for idx, group in enumerate(groups)
                            if group in groups_per_fold[i]]
            yield test_indices

def _sigmoid_calibration(df, y, sample_weight=None):
    """Probability Calibration with sigmoid method (Platt 2000)
    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.
    y : ndarray, shape (n_samples,)
        The targets.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.
    Returns
    -------
    a : float
        The slope.
    b : float
        The intercept.
    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    df = column_or_1d(df)
    y = column_or_1d(y)

    F = df  # F follows Platt's notations

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 1. / (prior0 + 2.)
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        P = expit(-(AB[0] * F + AB[1]))
        loss = -(xlogy(T, P) + xlogy(T1, 1. - P))
        if sample_weight is not None:
            return (sample_weight * loss).sum()
        else:
            return loss.sum()

    def grad(AB):
        # gradient of the objective function
        P = expit(-(AB[0] * F + AB[1]))
        TEP_minus_T1P = T - P
        if sample_weight is not None:
            TEP_minus_T1P *= sample_weight
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    
    return AB_[0], AB_[1]
