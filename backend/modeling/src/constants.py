import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

FORECAST_SIZE = 5
WINDOW_SIZE = FORECAST_SIZE * 10
TARGET = ["nbr_travels"]
COLUMNS_TO_SCALE = ["nbr_travels", "nbr_late_trains"]
NUM_FEATURES = len(TARGET)
MLRUNS_DIR = "/workdir/logs/mlruns"
FILE_PATH = "/workdir/src/sncfisajoke_user_tweets_new.xlsx"
SCALER_NAME = "standard"

MONTHS_TO_NUMBERS = {
    "janvier": "01",
    "février": "02",
    "mars": "03",
    "avril": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07",
    "août": "08",
    "septembre": "09",
    "octobre": "10",
    "novembre": "11",
    "décembre": "12",
}

model_config = {
    "gru_neurons": 128,
    "dropout": 0.2,
    "patience": 30,
    "learning_rate": 0.001,
    "batch_size": 64,
    "max_epochs": 500,
    "verbose": 0,
    "loss": tf.losses.MeanSquaredError(),
}
