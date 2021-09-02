import argparse
import json

from src.constants import (COLUMNS_TO_SCALE, FORECAST_SIZE, NUM_FEATURES,
                           SCALER_NAME, TARGET, WINDOW_SIZE)
from src.extract_data import load_data_from_db, process_data
from src.metrics import compute_metrics
from src.models.select_model import get_model
from src.scaler.select_scaler import get_scaler
from src.utils import build_time_series, split_train_val_test

import mlflow

mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
mlflow.tensorflow.autolog()


def main(model_name):
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("forcast_size", FORECAST_SIZE)
        mlflow.log_param("window_size", WINDOW_SIZE)
        mlflow.log_param("num_features", NUM_FEATURES)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("columns_to_scale", COLUMNS_TO_SCALE)
        mlflow.log_param("scaler_name", SCALER_NAME)

        df = load_data_from_db()
        df = process_data(df, use_covariates=False, use_manual_fill=True)

        train_df, val_df, test_df = split_train_val_test(df, split_size_val=0.3, split_size_test=0.1)
        scaler = get_scaler(scaler_name=SCALER_NAME, columns_to_scale=COLUMNS_TO_SCALE, target=TARGET)
        train_df = scaler.fit_transform(train_df)

        with open("/tmp/scaler.json", "w") as f:
            json.dump(scaler.get_config(), f)
            f.seek(0)
            mlflow.log_artifact(f.name)

        val_df = scaler.transform(val_df)
        test_df = scaler.transform(test_df)

        train_x, train_y = build_time_series(
            data=train_df, target=TARGET, window_size=WINDOW_SIZE, forecast_size=FORECAST_SIZE
        )
        val_x, val_y = build_time_series(
            data=val_df, target=TARGET, window_size=WINDOW_SIZE, forecast_size=FORECAST_SIZE
        )
        test_x, test_y = build_time_series(
            data=test_df, target=TARGET, window_size=WINDOW_SIZE, forecast_size=FORECAST_SIZE
        )
        model = get_model(
            model_name=model_name,
            target=TARGET,
            forecast_size=FORECAST_SIZE,
            num_features=NUM_FEATURES,
        )

        model.fit(train_x, train_y, val_x, val_y)
        preds = model.predict(test_x)
        preds = scaler.inverse_transform(preds)
        test_y = scaler.inverse_transform(test_y)

        mape, mse, mae = compute_metrics(test_y, preds)
        mlflow.log_metric("test_mape", mape * 100)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", default="lstm", type=str, help="Choose a model 'baseline' or 'lstm' ")
    args = parser.parse_args()
    print(args.model_name.upper())
    main(model_name=args.model_name)
