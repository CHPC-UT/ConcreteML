"""
SODNN_SM5
---------
A general script demonstrating nested cross-validation (often referred to as SM5)
for a single-output deep neural network regression task (SODNN). Uses:

- PyTorch for building and training an MLP (Net)
- Optuna for hyperparameter tuning
- An outer k-fold split for test sets
- An inner k-fold split for hyperparameter tuning (CV)

Author: Your Name or Organization
"""
import os
import numpy as np
import pandas as pd
import torch
import optuna
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error as mae

# If needed, append path to locate your dnntorch package
# sys.path.append("path/to/dnntorch")

from dnntorch import Net
import SODNN_SM5_Settings as settings

# ----------------------------- USER CONFIG -----------------------------
DATASET_CSV_PATH = settings.DATASET_CSV_PATH  
TARGET_COLUMN = settings.TARGET_COLUMN  
Other_target_columns = settings.OTHER_TARGET_COLUMNS
STATE= settings.STATE  
NAME_OF_THIS_STUDY = settings.NAME_OF_THIS_STUDY
OUTER_N_SPLITS = settings.OUTER_N_SPLITS
INNER_N_SPLITS = settings.INNER_N_SPLITS
OPTUNA_NUM_TRIALS = settings.OPTUNA_NUM_TRIALS
RANDOM_STATE_SEED = settings.RANDOM_STATE_SEED
EARLY_STOPPING_PATIENCE = settings.EARLY_STOPPING_PATIENCE
EARLY_STOPPING_MIN_DELTA = settings.EARLY_STOPPING_MIN_DELTA
MAX_EPOCH = settings.MAX_EPOCH


# -------------------------- LOAD & PREPARE DATA -------------------------
"""
Customize the data loading as needed. For example:
df = pd.read_csv(DATASET_CSV_PATH)
Ensure the file contains a column named TARGET_COLUMN for the single output.

We drop the target column from the features; no other columns are dropped 
since we want to keep them all (except the target).
"""

df = pd.read_csv(DATASET_CSV_PATH)
df = df.sample(frac=1, random_state=RANDOM_STATE_SEED).reset_index(drop=True)
# Remove any unwanted columns (if needed)
if Other_target_columns:
    for col in Other_target_columns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
assert TARGET_COLUMN in df.columns, f"{TARGET_COLUMN} not found in CSV columns."

X_all = df.drop(columns=[TARGET_COLUMN])
y_all = df[TARGET_COLUMN]

# Optionally, you can hold out a separate final test set (unrelated to outer folds).
# For example:
# df, df_final = train_test_split(df, test_size=10, random_state=RANDOM_STATE_SEED)
# Then use df_final for an ultimate final check, outside of nested CV.

# ------------------------ SETUP OUTER FOLDS -----------------------------
outer_kfold = KFold(n_splits=OUTER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE_SEED)
outer_folds = []
for train_idx, test_idx in outer_kfold.split(X_all):
    X_train_fold = X_all.iloc[train_idx]
    y_train_fold = y_all.iloc[train_idx]
    X_test_fold  = X_all.iloc[test_idx]
    y_test_fold  = y_all.iloc[test_idx]
    outer_folds.append((X_train_fold, y_train_fold, X_test_fold, y_test_fold))


# ------------------- HELPER FUNCTION FOR OPTUNA -------------------------
def nestedcv_optuna(fold_data, num_trials, study_label):
    """
    Performs hyperparameter tuning on the given 'fold_data' (inner cross-validation).

    fold_data: list of (X_train, y_train, X_val, y_val) for each inner fold
    num_trials: integer, how many hyperparameter trial runs to perform
    study_label: a string name for the study, e.g. "SODNN_SM5_study_fold_0"
    """

    def objective(trial, fold_list):
        # Determine input & output dimensions from the first fold
        input_dim = fold_list[0][0].shape[1]
        output_dim = 1  # single-output

        # Hyperparameter search space
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)
        list_num_neurons = [
            trial.suggest_int(f"num_neurons_layer{i}", 1, 128)
            for i in range(num_hidden_layers)
        ]
        activation_function = trial.suggest_categorical(
            "activation_function_layer", ["relu", "leaky_relu", "sigmoid", "tanh"]
        )
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1, log=True)
        dropout = trial.suggest_categorical("dropout", [0.0, 0.2, 0.5, 0.8])
        mini_batch_size = trial.suggest_categorical("mini_batch_size", [0, 16, 32])
        optimizer_name = trial.suggest_categorical("optimizer", ["ADAM", "SGD"])

        preds_cv = []
        reals_cv = []

        # Inner CV loop
        for (Xtr, ytr, Xval, yval) in fold_list:
            # Scale features
            scaler = StandardScaler()
            Xtr_scaled = scaler.fit_transform(Xtr)
            Xval_scaled = scaler.transform(Xval)

            Xtr_t  = torch.FloatTensor(Xtr_scaled)
            ytr_t  = torch.FloatTensor(ytr.values).view(-1, 1)
            Xval_t = torch.FloatTensor(Xval_scaled)
            yval_t = torch.FloatTensor(yval.values).view(-1, 1)

            # Build model
            model = Net(
                num_hidden_layers,
                list_num_neurons,
                activation_function,
                dropout,
                in_features=input_dim,
                out_features=output_dim,
            )

            # Train
            model.train_(
                optimizer_learning_rate=learning_rate,
                optimizer_weight_decay=weight_decay,
                early_stopper_patience=EARLY_STOPPING_PATIENCE,
                early_stopper_min_delta=EARLY_STOPPING_MIN_DELTA,
                max_epoch=MAX_EPOCH,
                mini_batch_size=mini_batch_size,
                X_train=Xtr_t,
                y_train=ytr_t,
                optimizer=optimizer_name,
            )

            # Predict on validation
            y_pred_val = model.predict(Xval_t)
            preds_cv.extend(y_pred_val)
            reals_cv.extend(yval_t)

        # Evaluate R^2 across entire inner CV
        try:
            return r2_score(reals_cv, preds_cv)
        except:
            return np.nan

    # Setup persistent Optuna storage
    if not os.path.exists(NAME_OF_THIS_STUDY):
        os.makedirs(NAME_OF_THIS_STUDY)
    storage_path = f"sqlite:///{os.path.join(NAME_OF_THIS_STUDY, NAME_OF_THIS_STUDY)}.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        study_name=study_label,
        storage=storage_path,
        load_if_exists=True,
    )

    # If trials already exist, only run the remainder
    try:
        existing = study.trials_dataframe().shape[0]
    except:
        existing = 0
    needed = num_trials - existing
    if needed < 0:
        needed = 0

    # Run the optimization
    study.optimize(lambda t: objective(t, fold_data), n_trials=needed)
    print("Best R^2 so far for", study_label, ":", study.best_value)


# --------------------- EXECUTION (TRAIN / TEST) --------------------------

if 'train' in STATE:
    # Outer loop: for each fold, do an inner CV with Optuna for hyperparams
    for i, (X_train_full, y_train_full, X_test_fold, y_test_fold) in enumerate(outer_folds):
        # Build inner folds
        inner_kf = KFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE_SEED)
        fold_for_optuna = []
        for train_idx, val_idx in inner_kf.split(X_train_full):
            X_tr = X_train_full.iloc[train_idx]
            y_tr = y_train_full.iloc[train_idx]
            X_val = X_train_full.iloc[val_idx]
            y_val = y_train_full.iloc[val_idx]
            fold_for_optuna.append((X_tr, y_tr, X_val, y_val))

        study_label_fold = f"{NAME_OF_THIS_STUDY}_outerfold_{i}"
        nestedcv_optuna(fold_for_optuna, OPTUNA_NUM_TRIALS, study_label_fold)

if 'test' in STATE:
    # Ensemble approach: each inner model (from K inner folds) predicts the outer test.
    # The predictions from all inner models are averaged to yield the final outer-test predictions.

    # Lists for storing predictions/targets across folds for potential overall metrics
    Real_test_set = []
    Pred_test_set = []
    Real_train_set = []
    Pred_train_set = []
    Real_cv_set = []
    Pred_cv_set = []

    for i, (X_train_full, y_train_full, X_test_fold, y_test_fold) in enumerate(outer_folds):
        study_label_fold = f"{NAME_OF_THIS_STUDY}_outerfold_{i}"
        storage_path = f"sqlite:///{os.path.join(NAME_OF_THIS_STUDY, NAME_OF_THIS_STUDY)}.db"

        # 1) Load the best hyperparameters found for this outer fold
        study = optuna.load_study(study_name=study_label_fold, storage=storage_path)
        best_params = study.best_params
        best_value = study.best_value
        print(f"Outer fold {i}: best R^2 from inner CV was {best_value:.4f}")

        # Unpack hyperparameters
        num_hidden_layers = best_params["num_hidden_layers"]
        list_num_neurons = [best_params[f"num_neurons_layer{layer_idx}"]
                            for layer_idx in range(num_hidden_layers)]
        activation_function = best_params["activation_function_layer"]
        learning_rate = best_params["learning_rate"]
        weight_decay = best_params["weight_decay"]
        dropout = best_params["dropout"]
        mini_batch_size = best_params["mini_batch_size"]
        optimizer_name = best_params["optimizer"]

        input_dim = X_train_full.shape[1]  # single-output
        output_dim = 1

        # 2) For each outer fold, re-run the SAME inner K-fold
        inner_kf = KFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE_SEED)

        # We'll accumulate outer-test predictions from each inner model
        test_preds_accumulator = []  # shape will be [K, num_test_samples]

        # Convert outer test fold to NumPy for consistent scaling
        X_test_fold_arr = X_test_fold.to_numpy()
        y_test_fold_arr = y_test_fold.to_numpy()

        Real_cv_this_fold = []
        Pred_cv_this_fold = []
        for train_idx_inner, val_idx_inner in inner_kf.split(X_train_full):
            # Extract inner-train and inner-val
            X_tr = X_train_full.iloc[train_idx_inner]
            y_tr = y_train_full.iloc[train_idx_inner]
            X_val = X_train_full.iloc[val_idx_inner]
            y_val = y_train_full.iloc[val_idx_inner]

            # Scale on inner-train, transform inner-val and outer-test
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr.to_numpy())
            X_val_scaled = scaler.transform(X_val.to_numpy())
            X_test_scaled = scaler.transform(X_test_fold_arr)

            # Convert to tensors
            X_tr_t = torch.FloatTensor(X_tr_scaled).view(-1, input_dim)
            y_tr_t = torch.FloatTensor(y_tr.values).view(-1, 1)

            X_val_t = torch.FloatTensor(X_val_scaled).view(-1, input_dim)
            y_val_t = torch.FloatTensor(y_val.values).view(-1, 1)

            X_test_t = torch.FloatTensor(X_test_scaled).view(-1, input_dim)

            # Build & train a model on this inner-train portion
            model = Net(
                num_hidden_layers,
                list_num_neurons,
                activation_function,
                dropout,
                in_features=input_dim,
                out_features=output_dim,
            )
            model.train_(
                optimizer_learning_rate=learning_rate,
                optimizer_weight_decay=weight_decay,
                early_stopper_patience=EARLY_STOPPING_PATIENCE,
                early_stopper_min_delta=EARLY_STOPPING_MIN_DELTA,
                max_epoch=MAX_EPOCH,
                mini_batch_size=mini_batch_size,
                X_train=X_tr_t,
                y_train=y_tr_t,
                optimizer=optimizer_name,
            )

            # Predict on the outer test set with this inner model
            fold_test_preds = model.predict(X_test_t)  # shape: [num_test_samples, 1]
            test_preds_accumulator.append(fold_test_preds.ravel())

            # Optionally gather predictions on the inner validation set
            val_preds = model.predict(X_val_t)  # shape: [num_val_samples]
            Real_cv_set.extend(y_val_t.cpu().numpy().ravel())
            Pred_cv_set.extend(val_preds.ravel())

            # Optionally gather predictions on the inner training set
            train_preds = model.predict(X_tr_t)  # shape: [num_train_samples]
            Real_train_set.extend(y_tr_t.cpu().numpy().ravel())
            Pred_train_set.extend(train_preds.ravel())

            # fill Real_cv_this_fold and Pred_cv_this_fold with the current fold's values
            Real_cv_this_fold.extend(y_val_t.cpu().numpy().ravel())
            Pred_cv_this_fold.extend(val_preds.ravel())
        
        # Check whether we take a same result as training set or not
        r2_score_cv_this_fold = r2_score(Real_cv_this_fold, Pred_cv_this_fold)
        if r2_score_cv_this_fold != best_value:
            print(f"Warning: R^2 for inner CV fold {i} does not match best value from Optuna.")

        # 3) Average the outer-test predictions from the K inner models
        test_preds_accumulator = np.array(test_preds_accumulator)  # shape: [K, num_test_samples]
        mean_test_preds = np.mean(test_preds_accumulator, axis=0)  # shape: [num_test_samples]

        # 4) Store these final test predictions for the outer fold
        Real_test_set.extend(y_test_fold_arr)
        Pred_test_set.extend(mean_test_preds)

    # 5) After all outer folds, compute final aggregated metrics

    # Outer test metrics (across all folds)
    test_r2  = r2_score(Real_test_set, Pred_test_set)
    test_rmse = mean_squared_error(Real_test_set, Pred_test_set)**0.5
    test_mae  = mae(Real_test_set, Pred_test_set)

    print("=== Final Outer Test Set Metrics (Averaged Ensemble) ===")
    print(f"R^2:  {test_r2:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE:  {test_mae:.4f}")

    # Inner CV metrics (if you want a sense of how well the inner models performed on their val sets)
    cv_r2  = r2_score(Real_cv_set, Pred_cv_set)
    cv_rmse = mean_squared_error(Real_cv_set, Pred_cv_set)**0.5
    cv_mae  = mae(Real_cv_set, Pred_cv_set)

    print("=== Final Inner CV Set Metrics (Across All Inner Folds) ===")
    print(f"R^2:  {cv_r2:.4f}")
    print(f"RMSE: {cv_rmse:.4f}")
    print(f"MAE:  {cv_mae:.4f}")

    # Inner train metrics (aggregated from each inner model's training set)
    train_r2  = r2_score(Real_train_set, Pred_train_set)
    train_rmse = mean_squared_error(Real_train_set, Pred_train_set)**0.5
    train_mae  = mae(Real_train_set, Pred_train_set)

    print("=== Final Inner Train Set Metrics (Across All Inner Folds) ===")
    print(f"R^2:  {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAE:  {train_mae:.4f}")




else:
    print(f"Unknown STATE '{STATE}'. Use either 'train' or 'test'.")
