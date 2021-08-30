import pandas as pd
from src.scaler.base_scaler import BaseScaler


class StandardScaler(BaseScaler):
    def __init__(self, columns_to_scale: list, target: list):
        super().__init__(columns_to_scale, target)

    def fit_transform(self, data: pd.DataFrame):
        df = data.copy()
        for col in self.columns_to_scale:
            mean_column = df[col].mean()
            std_column = df[col].std()
            self.dic[col] = {
                "mean": mean_column,
                "std": std_column,
            }
            df[col] = (df[col] - mean_column) / std_column
        return df

    def transform(self, data: pd.DataFrame):
        df = data.copy()
        for col in self.columns_to_scale:
            df[col] = (df[col] - self.dic[col]["mean"]) / self.dic[col]["std"]
        return df

    def inverse_transform(self, data):
        list_inverse_scale_data = []
        for i in range(len(data)):
            x = data[i].copy()
            for col in self.columns_to_scale:
                if col in self.target:
                    x[col] = x[col] * self.dic[col]["std"] + self.dic[col]["mean"]
            list_inverse_scale_data.append(x)
        return list_inverse_scale_data
