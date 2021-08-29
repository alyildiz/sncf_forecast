from src.models.baseline import Baseline
from src.models.lstm_model import LSTMModel


def get_model(model_name: str, forecast_size: int, target: list, num_features: int):
    if model_name == "baseline":
        model = Baseline(forecast_size=forecast_size, target=target)
    elif model_name == "lstm":
        model = LSTMModel(forecast_size=forecast_size, num_features=num_features)
    else:
        raise KeyError("Available model : 'baseline', 'lstm' ")

    return model
