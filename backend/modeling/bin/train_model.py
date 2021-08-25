from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.collect_data import load_data_from_db
from src.constants import FORECAST_SIZE, MLRUNS_DIR, SPACE, WINDOW_SIZE
from src.utils import build_time_series

import mlflow
from shared.db.utils import get_mongo_client

mlflow.set_tracking_uri(MLRUNS_DIR)


def main():
    mongo_client = get_mongo_client()
    db = mongo_client["BTC"]

    mlflow.sklearn.autolog(max_tuning_runs=None)

    df_historical = load_data_from_db(db=db)

    target = df_historical["open"].tolist()
    x, y = build_time_series(array=target, window_size=WINDOW_SIZE, forecast_size=FORECAST_SIZE)
    tscv = TimeSeriesSplit(gap=FORECAST_SIZE - 1, n_splits=5, test_size=1)
    model = RandomForestRegressor(random_state=1)

    search = GridSearchCV(
        model,
        SPACE,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        verbose=1,
    )
    with mlflow.start_run() as run:
        result = search.fit(x, y)


if __name__ == "__main__":
    main()
