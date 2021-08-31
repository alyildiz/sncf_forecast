from abc import ABCMeta, abstractmethod

import pandas as pd


class BaseScaler(metaclass=ABCMeta):
    def __init__(self, columns_to_scale: list, target: list, dic: dict = {}) -> None:
        self.columns_to_scale = columns_to_scale
        self.dic = dic
        self.target = target

    def get_config(self):
        dic = {
            "columns_to_scale": self.columns_to_scale,
            "dic": self.dic,
            "target": self.target,
        }
        return dic

    @classmethod
    def load(cls, data):
        return cls(data["columns_to_scale"], data["target"], data["dic"])

    @abstractmethod
    def fit_transform(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def inverse_transform(self, data):
        pass
