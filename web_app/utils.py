import sqlite3
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

import mlflow

tick_size = 12
axis_title_size = 16

dic = {
    "loss": "Loss",
    "val_loss": "Validation Loss",
    "mean_squared_error": "MSE",
    "val_mean_squared_error": "Validation MSE",
    "mean_absolute_error": "MAE",
    "val_mean_absolute_error": "Validation MAE",
}


def preload_models(con):
    dic_models = {
        "lstm": {
            "run_uuid": get_best_model_today("lstm", con),
            "model": mlflow.keras.load_model(
                f'/workdir/artifacts/0/{get_best_model_today("lstm", con)}/artifacts/model'
            ),
        },
        "autoencoder": {
            "run_uuid": get_best_model_today("autoencoder", con),
            "model": mlflow.keras.load_model(
                f'/workdir/artifacts/0/{get_best_model_today("autoencoder", con)}/artifacts/model'
            ),
        },
    }
    return dic_models


def load_model(model_name, dic_models):
    if model_name == "LSTM":
        run_uuid = dic_models["lstm"]["run_uuid"]
        model = dic_models["lstm"]["model"]
    elif model_name == "Autoencoder":
        run_uuid = dic_models["autoencoder"]["run_uuid"]
        model = dic_models["autoencoder"]["model"]

    else:
        raise KeyError("Model not supported")

    return run_uuid, model


def get_best_model_today(model_name, con):
    today = date.today()
    runs = pd.read_sql_query("select * from runs", con)
    runs["end_time"] = pd.to_datetime(runs["end_time"], unit="ms")
    runs = runs.loc[
        (pd.to_datetime(today) < runs["end_time"]) & (runs["end_time"] < pd.to_datetime(today + timedelta(days=1)))
    ]
    # get the params
    params = pd.read_sql_query("select * from params", con)
    # left join and filter by model name
    merged = runs.merge(params, how="left", left_on="run_uuid", right_on="run_uuid")
    merged = merged.loc[merged["value"] == model_name]
    # get the metrics
    metrics = pd.read_sql_query("select * from metrics", con)
    # left join and filter by metrics
    merged = merged.merge(metrics, how="left", left_on="run_uuid", right_on="run_uuid", suffixes=("", "_metrics"))
    merged = merged.loc[merged["key_metrics"] == "test_mape"]
    merged = merged.sort_values(["value_metrics"], ascending=True)

    best_model_uuid = merged.iloc[0]["run_uuid"]
    return best_model_uuid


def get_model_param(run_uuid, con):
    params = pd.read_sql_query("select * from params", con)
    params = params.loc[params["run_uuid"] == run_uuid]
    window_size = params.loc[params["key"] == "window_size"]["value"].values[0]
    scaler_name = params.loc[params["key"] == "scaler_name"]["value"].values[0]
    forecast_size = params.loc[params["key"] == "forcast_size"]["value"].values[0]
    target = params.loc[params["key"] == "target"]["value"].values[0]
    model_name = params.loc[params["key"] == "model_name"]["value"].values[0]

    return (
        int(window_size),
        scaler_name,
        int(forecast_size),
        [val[1:-1] for val in target[1:-1].split(" ")],
        model_name,
    )


def plot_train_loss(run_uuid, list_metric, con, area, title):
    metrics = pd.read_sql_query("select * from metrics", con)
    metrics = metrics.loc[metrics["run_uuid"] == run_uuid]

    # plot metric
    metrics_filtered = metrics.loc[metrics["key"] == list_metric[0]][["step", "value"]]
    metrics_filtered = metrics_filtered.drop_duplicates().reset_index(drop=True)
    plot_metric = (
        alt.Chart(metrics_filtered)
        .encode(alt.X("step", title="Epochs"))
        .mark_line(color="#A7C7E7", size=4)
        .encode(
            y=alt.Y("value", axis=alt.Axis(title=dic[list_metric[0]])),
            tooltip=[alt.Tooltip("value", title=f"{list_metric[0]}")],
        )
    )

    # plot validation metric
    metrics_filtered = metrics.loc[metrics["key"] == list_metric[1]][["step", "value"]]
    metrics_filtered = metrics_filtered.drop_duplicates().reset_index(drop=True)
    plot_val_metric = (
        alt.Chart(metrics_filtered)
        .encode(alt.X("step", title="Epochs"))
        .mark_line(color="#FFAA33", size=4)
        .encode(
            y=alt.Y("value", axis=alt.Axis(title=dic[list_metric[1]])),
            tooltip=[alt.Tooltip("value", title=f"{list_metric[1]}")],
        )
    )

    fig = alt.layer(plot_metric, plot_val_metric).configure_axis(
        labelFontSize=tick_size, titleFontSize=axis_title_size
    )

    if st.util.env_util.is_repl():
        fig.save("nbr_travels_data.svg")

    area.subheader(title)
    area.altair_chart(fig, use_container_width=True)


if __name__ == "__main__":
    con = sqlite3.connect("/workdir/data/mlflow.db")
    cursor = con.cursor()
    # plot_train_loss("e669cbbb4f8545a09e935fd2cad24a24", ["loss", "val_loss"], con, "j")
    get_model_param("e669cbbb4f8545a09e935fd2cad24a24", con)
