import os

from src.models.lstm_model import LSTMModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


class BasicLSTM(LSTMModel):
    def __init__(self, forecast_size, num_features, model_config, target):
        super().__init__(forecast_size, num_features, model_config, target)

    def _build_and_compile(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(self.model_config["gru_neurons"], return_sequences=False),
                tf.keras.layers.Dropout(self.model_config["dropout"]),
                tf.keras.layers.Dense(self.forecast_size * self.num_features),
                tf.keras.layers.Reshape([self.forecast_size, self.num_features]),
            ]
        )

        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.model_config["patience"], mode="min"
        )

        model.compile(
            loss=self.model_config["loss"],
            optimizer=tf.optimizers.Adam(learning_rate=self.model_config["learning_rate"]),
            metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredError()],
        )

        return model
