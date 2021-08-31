import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402

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
