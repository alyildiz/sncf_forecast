import pandas as pd
from pandas.core.frame import DataFrame


class Scaler:
    def __init__(self, columns_to_scale: list) -> None:
        self.columns_to_scale = columns_to_scale
        self.dic_mean_std = {}

    def fit_transform(self, data: pd.DataFrame):
        df = data.copy()
        for column in self.columns_to_scale:
            mean_column = df[column].mean()
            std_column = df[column].std()
            self.dic_mean_std[column] = {
                "mean": mean_column,
                "std": std_column,
            }
            df[column] = (df[column] - mean_column) / std_column
        return df

    def transform(self, data: pd.DataFrame):
        df = data.copy()
        for column in self.columns_to_scale:
            df[column] = (df[column] - self.dic_mean_std[column]["mean"]) / self.dic_mean_std[column]["std"]
        return df
