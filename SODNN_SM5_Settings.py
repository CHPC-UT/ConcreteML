# Path to the CSV file
DATASET_CSV_PATH = "concrete.csv"  
TARGET_COLUMN = "flow"  # Name of the target column in your CSV
OTHER_TARGET_COLUMNS = ['slump', 'compressive_strength']  # List of columns to drop

#####################################################################

# States to be used in the code
STATE = ["train", 'test']  # Specifies whether to run training or testing
# If you want to run both, set STATE = ["train", "test"].

#####################################################################

# Name of the study
NAME_OF_THIS_STUDY = "SODNN_SM5_study"  # Choose a name for your study
# This will be used to save the Optuna study and results.

#####################################################################

# Parameters for nested cross-validation
OUTER_N_SPLITS = 10  # Number of splits for outer fold
INNER_N_SPLITS = 5   # Number of splits for inner fold

#####################################################################

# Number of trials for Optuna optimization
OPTUNA_NUM_TRIALS = 10

#####################################################################

# Random seed for reproducibility
RANDOM_STATE_SEED = 42  

#####################################################################

# Early stopping parameters for training
EARLY_STOPPING_PATIENCE = 50  # Number of epochs with no improvement after which training will stop
EARLY_STOPPING_MIN_DELTA = 1  # Minimum change to qualify as an improvement
MAX_EPOCH = 2000  # Maximum number of epochs to train the model

#####################################################################

# Print a warning message to notify the user to check hyperparameter ranges
print("Warning: Please check the hyperparameter ranges in the code.")
