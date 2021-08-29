import pandas as pd
from src.constants import FILE_PATH
from src.extract_data import load_data, process_data
from src.metrics import compute_metrics
from src.models.select_model import get_model
from src.scaler import Scaler
from src.utils import build_time_series, split_train_val_test

FORECAST_SIZE = 5
WINDOW_SIZE = FORECAST_SIZE * 10
TARGET = ["nbr_travels"]
NUM_FEATURES = len(TARGET)


def main():
    df = load_data(FILE_PATH)
    df = process_data(df, use_covariates=True)
    train_df, val_df, test_df = split_train_val_test(df, split_size_val=0.3, split_size_test=0.1)
    scaler = Scaler(columns_to_scale=["nbr_travels", "nbr_late_trains"])
    train_df = scaler.fit_transform(train_df)
    val_df = scaler.transform(val_df)
    test_df = scaler.transform(test_df)

    train_x, train_y = build_time_series(
        data=train_df, target=TARGET, window_size=WINDOW_SIZE, forecast_size=FORECAST_SIZE
    )
    val_x, val_y = build_time_series(data=val_df, target=TARGET, window_size=WINDOW_SIZE, forecast_size=FORECAST_SIZE)
    test_x, test_y = build_time_series(
        data=test_df, target=TARGET, window_size=WINDOW_SIZE, forecast_size=FORECAST_SIZE
    )
    model = get_model(
        model_name="baseline",
        target=TARGET,
        forecast_size=FORECAST_SIZE,
        num_features=NUM_FEATURES,
    )

    model.fit(train_x, train_y, val_x, val_y)
    preds = model.predict(test_x)
    compute_metrics(test_y, preds)


if __name__ == "__main__":
    main()
