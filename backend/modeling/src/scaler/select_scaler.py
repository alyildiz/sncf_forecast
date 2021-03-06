from src.scaler.min_max_scaler import MinMaxScaler
from src.scaler.standard_scaler import StandardScaler


def get_scaler(scaler_name: str, target, columns_to_scale):
    if scaler_name == "standard":
        scaler = StandardScaler(target=target, columns_to_scale=columns_to_scale)

    elif scaler_name == "min_max":
        scaler = MinMaxScaler(target=target, columns_to_scale=columns_to_scale)

    else:
        raise KeyError("Available scaler : 'standard', 'min_max' ")

    return scaler


def select_scaler_cls(scaler_name, data):
    if scaler_name == "standard":
        scaler = StandardScaler.load(data)
    elif scaler_name == "min_max":
        scaler = MinMaxScaler.load(data)
    else:
        raise KeyError("Scaler not supported.")
    return scaler
