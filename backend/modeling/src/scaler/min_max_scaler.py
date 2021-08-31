import pandas as pd
from src.scaler.base_scaler import BaseScaler


class MinMaxScaler(BaseScaler):
    def __init__(self, columns_to_scale: list, target: list, dic: dict = {}):
        super().__init__(columns_to_scale, target, dic)

    def fit_transform(self, data: pd.DataFrame):
        df = data.copy()
        for col in self.columns_to_scale:
            min_column = df[col].min()
            max_column = df[col].max()
            self.dic[col] = {
                "min": min_column,
                "max": max_column,
            }
            df[col] = (df[col] - min_column) / (max_column - min_column)
        return df

    def transform(self, data: pd.DataFrame):
        df = data.copy()
        for col in self.columns_to_scale:
            df[col] = (df[col] - self.dic[col]["min"]) / (self.dic[col]["max"] - self.dic[col]["min"])
        return df

    def inverse_transform(self, data):
        list_inverse_scale_data = []
        for i in range(len(data)):
            x = data[i].copy()
            for col in self.columns_to_scale:
                if col in self.target:
                    x[col] = x[col] * (self.dic[col]["max"] - self.dic[col]["min"]) + self.dic[col]["min"]
            list_inverse_scale_data.append(x)
        return list_inverse_scale_data
