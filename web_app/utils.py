from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st
from src.extract_data import load_data_from_db, process_data
from src.inference import inference

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


def setup_page():
    alt.renderers.set_embed_options(scaleFactor=2)

    st.set_page_config(layout="wide")
    st.title("French railways : daily forecast of running trains")
    st.markdown(
        """*Check out the github
        [here](https://github.com/alyildiz/btc_forecast) and the
        daily data [here](https://twitter.com/sncfisajoke) .*"""
    )

    st.sidebar.title("Control Panel")
    # User inputs on the control panel
    st.sidebar.subheader("Model selection")
    model_name_webapp = st.sidebar.selectbox(
        "Choose the model",
        ["LSTM", "Autoencoder", "Baseline"],
        help="The forecast is model dependant.",
    )
    return model_name_webapp


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
        "baseline": {
            "run_uuid": get_best_model_today("baseline", con),
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
    elif model_name == "Baseline":
        run_uuid = dic_models["baseline"]["run_uuid"]
        model = None
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


def get_test_metric(run_uuid, con):
    metrics = pd.read_sql_query("select * from metrics", con)
    metrics = metrics.loc[metrics["run_uuid"] == run_uuid]
    test_mape = metrics.loc[metrics["key"] == "test_mape"]["value"].values[0]
    test_mae = metrics.loc[metrics["key"] == "test_mae"]["value"].values[0]
    test_mse = metrics.loc[metrics["key"] == "test_mse"]["value"].values[0]

    return str(round(float(test_mape), 2)) + "%", round(float(test_mae), 2), round(float(test_mse), 2)


def get_end_time(run_uuid, con):
    runs = pd.read_sql_query("select * from runs", con)
    runs = runs.loc[runs["run_uuid"] == run_uuid]
    end_time = runs["end_time"].values[0]

    return pd.to_datetime(end_time, unit="ms")


def plot_timeline(inputs, preds, model_name_webapp, area):
    inputs = inputs.reset_index()
    preds = preds.astype(int).reset_index()

    base = alt.Chart(inputs).transform_calculate(
        Inputs="'Inputs'",
        Forecast="'Forecast'",
    )
    scale = alt.Scale(domain=["Inputs", "Forecast"], range=["#A7C7E7", "#FFAA33"])

    inputs_plot = base.mark_line(
        size=4,
        point={
            "filled": False,
            "fill": "white",
            "size": 64,
            "color": "#A7C7E7",
        },
    ).encode(
        x=alt.X("date:T", title="Days"),
        y=alt.Y("nbr_travels", title="Number of trains running"),
        color=alt.Color("Inputs:N", scale=scale, title=""),
        tooltip=[alt.Tooltip("nbr_travels", title="Number of trains"), alt.Tooltip("date:T", title="Day")],
    )

    base2 = alt.Chart(preds).transform_calculate(
        Inputs="'Inputs'",
        Forecast="'Forecast'",
    )

    preds_plot = base2.mark_line(
        size=4,
        point={
            "filled": False,
            "fill": "white",
            "size": 64,
            "color": "#FFAA33",
        },
    ).encode(
        x=alt.X("index:T", title="Day"),
        y=alt.Y("nbr_travels", title="Number of trains running"),
        color=alt.Color("Forecast:N", scale=scale, title=""),
        tooltip=[alt.Tooltip("nbr_travels", title="Number of trains"), alt.Tooltip("index:T", title="Day")],
    )

    fig = alt.layer(inputs_plot, preds_plot).configure_axis(labelFontSize=tick_size, titleFontSize=axis_title_size)

    if st.util.env_util.is_repl():
        fig.save("nbr_travels_data.svg")

    area.subheader(f"Current data and forecast using {model_name_webapp}")
    area.altair_chart(fig, use_container_width=True)


def display_metrics(run_uuid, mape, mae, mse, area):
    area.subheader("Performance over test set")
    area.markdown(f"**MAPE : {mape}**")
    area.markdown(f"**MAE : {mae}**")
    area.markdown(f"**MSE : {mse}**")
    area.markdown(f"[Model configuration](http://localhost:5000/#/experiments/0/runs/{run_uuid})")


def get_timeline(model, scaler, window_size):
    inputs = load_data_from_db()
    inputs = process_data(inputs, use_covariates=False)
    inputs = inputs.iloc[-window_size:, :]
    preds = inference(model, scaler, inputs)

    return inputs, preds


if __name__ == "__main__":
    pass
