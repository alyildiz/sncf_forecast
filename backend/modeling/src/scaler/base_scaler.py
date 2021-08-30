from abc import ABCMeta, abstractmethod

import pandas as pd


class BaseScaler(metaclass=ABCMeta):
    def __init__(self, columns_to_scale: list, target: list) -> None:
        self.columns_to_scale = columns_to_scale
        self.dic = {}
        self.target = target

    @abstractmethod
    def fit_transform(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def inverse_transform(self, data):
        pass
