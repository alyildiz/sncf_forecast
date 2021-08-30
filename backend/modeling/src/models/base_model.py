from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, forecast_size, target):
        self.forecast_size = forecast_size
        self.target = target

    @abstractmethod
    def predict(self):
        pass
