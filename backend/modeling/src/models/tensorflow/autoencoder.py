import os

from src.models.lstm_model import LSTMModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf  # noqa: E402


class Autoencoder(LSTMModel):
    def __init__(self, forecast_size=None, num_features=None, model_config=None, target=None, model=None):
        super().__init__(forecast_size, num_features, model_config, target, model)

    @classmethod
    def load(cls, model, forecast_size, target):
        return cls(model=model, forecast_size=forecast_size, target=target)

    def _build_and_compile(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(256, activation="relu"),
                tf.keras.layers.RepeatVector(self.forecast_size * self.num_features),
                tf.keras.layers.LSTM(256, activation="relu", return_sequences=True),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
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
