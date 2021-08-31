from src.hp_search import model_config
from src.models.baseline import Baseline
from src.models.tensorflow.autoencoder import Autoencoder
from src.models.tensorflow.lstm import BasicLSTM


def get_model(model_name: str, forecast_size: int, target: list, num_features: int):
    if model_name == "baseline":
        model = Baseline(forecast_size=forecast_size, target=target)
    elif model_name == "autoencoder":
        model = Autoencoder(
            forecast_size=forecast_size, num_features=num_features, model_config=model_config, target=target
        )
    elif model_name == "lstm":
        model = BasicLSTM(
            forecast_size=forecast_size, num_features=num_features, model_config=model_config, target=target
        )
    else:
        raise KeyError("Available model : 'baseline', 'lstm' ")

    return model


def select_model_cls(model_name, model, forecast_size, target):
    if model_name == "baseline":
        model = Baseline(forecast_size=forecast_size, target=target)
    elif model_name == "autoencoder":
        model = Autoencoder.load(model, forecast_size, target)
    elif model_name == "lstm":
        model = BasicLSTM.load(model, forecast_size, target)
    else:
        raise KeyError("Available model : 'baseline', 'lstm', 'autoencoder'")

    return model
