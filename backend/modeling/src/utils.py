import os

import finnhub
import numpy as np


def get_finnhub_client():
    return finnhub.Client(api_key=os.environ["API_KEY"])


def build_time_series(array: np.ndarray, window_size: int, forecast_size: int):
    list_x, list_y = [], []
    for i in range(0, len(array) - forecast_size - window_size + 1):
        x = array[i: i + window_size]
        y = array[i + window_size: i + window_size + forecast_size]
        list_x.append(x)
        list_y.append(y)

    return np.array(list_x), np.array(list_y)
