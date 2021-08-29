from datetime import timedelta

import numpy as np
import pandas as pd
from src.models.base_model import BaseModel


class Baseline(BaseModel):
    def __init__(self, forecast_size, target):
        super().__init__(forecast_size)
        self.target = target

    def fit(self, train_x, train_y, val_x, val_y):
        print("\nTraining not required for Baseline model.")
        pass

    def predict(self, test_x):
        if type(test_x) == list:
            preds = []
            for batch in test_x:
                pred = self._make_single_pred(batch)
                preds.append(pred)

        elif type(test_x) == pd.DataFrame:
            preds = self._make_single_pred(test_x)

        return np.array(preds)

    def _make_single_pred(self, batch):
        data = batch.copy()
        data["day"] = data.index
        last_day = data["day"].iloc[-1]
        data["day"] = data.day.apply(lambda x: x.strftime("%A"))

        first_forecast_day = last_day + timedelta(days=1)
        date_range = pd.date_range(start=first_forecast_day, periods=self.forecast_size)
        date_range = [x.strftime("%A") for x in date_range]
        pred = pd.DataFrame(date_range, columns=["days"])

        pred = pred.merge(data.groupby("day").mean(), how="left", left_on="days", right_index=True)[self.target].values

        return pred
