from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, forecast_size):
        self.forecast_size = forecast_size
        pass

    @abstractmethod
    def predict(self):
        pass
