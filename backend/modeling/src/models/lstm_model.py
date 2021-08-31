import os
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from src.models.base_model import BaseModel

tf.random.set_seed(0)
import mlflow


class LSTMModel(BaseModel):
    def __init__(self, forecast_size, num_features, model_config, target, model):
        super().__init__(forecast_size, target)
        self.num_features = num_features
        self.model_config = model_config
        if model is None:
            self.model = self._build_and_compile()
        else:
            self.model = model

    def fit(
        self,
        train_x: List[pd.DataFrame],
        train_y: List[pd.DataFrame],
        val_x: List[pd.DataFrame],
        val_y: List[pd.DataFrame],
    ):
        train_x = np.array([x.values for x in train_x])
        train_y = np.array([x.values for x in train_y])

        val_x = np.array([x.values for x in val_x])
        val_y = np.array([x.values for x in val_y])
        print("\nTraining model...\n")

        history = self.model.fit(
            train_x,
            train_y,
            batch_size=self.model_config["batch_size"],
            epochs=self.model_config["max_epochs"],
            validation_data=(val_x, val_y),
            callbacks=[CustomCallback(), self.early_stopping],
            verbose=self.model_config["verbose"],
        )

        return history

    def predict(self, test_x):
        if type(test_x) == list:
            test_x_array = np.array([x.values for x in test_x])
            preds_numpy = self.model(test_x_array).numpy()
            preds = []
            for i, df in enumerate(test_x):
                pred = self._format_pred(df, preds_numpy, i)
                preds.append(pred)

        elif type(test_x) == pd.DataFrame:
            test_x_array = np.expand_dims(test_x.values, axis=0)
            preds_numpy = self.model(test_x_array).numpy()
            preds = self._format_pred(test_x, preds_numpy, 0)

        else:
            TypeError("Type not supported.")

        return preds

    def _format_pred(self, inputs, pred, i):
        last_day = inputs.index[-1]
        first_forecast_day = last_day + timedelta(days=1)
        date_range = pd.date_range(start=first_forecast_day, periods=self.forecast_size)
        pred = pd.DataFrame(pred[i, :, :], index=date_range, columns=self.target)
        return pred


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(logs, step=epoch)
