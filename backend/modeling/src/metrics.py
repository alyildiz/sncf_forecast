import numpy as np
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)


def compute_metrics(test_y: list, preds: np.ndarray, display: bool = True):
    test_y = np.array([x.values for x in test_y])
    preds = np.array([x.values for x in preds])
    nsamples, nx, ny = test_y.shape

    test_y = test_y.reshape((nsamples, nx * ny))
    preds = preds.reshape((nsamples, nx * ny))

    mape = mean_absolute_percentage_error(test_y, preds)
    mse = mean_squared_error(test_y, preds)
    mae = mean_absolute_error(test_y, preds)

    if display:
        print(f"MAPE : {round(mape*100, 2)} %")
        print(f"MSE : {round(mse, 2)}")
        print(f"MAE : {round(mae, 2)}")

    return mape, mse, mae
