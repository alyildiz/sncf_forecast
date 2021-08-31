import json
import sqlite3

import altair as alt
import pandas as pd
import streamlit as st
from src.extract_data import load_data_from_db, process_data
from src.inference import inference
from src.models.select_model import select_model_cls
from src.scaler.select_scaler import select_scaler_cls
from utils import get_model_param, load_model, plot_train_loss, preload_models

from shared.db.utils import get_mongo_client

client = get_mongo_client()

con = sqlite3.connect("/workdir/data/mlflow.db")
cursor = con.cursor()

alt.renderers.set_embed_options(scaleFactor=2)

st.set_page_config(layout="wide")
st.title("French railways : daily forecast of running trains")
st.markdown(
    """*Check out the github
    [here](https://github.com/alyildiz/btc_forecast) and the daily data [here](https://twitter.com/sncfisajoke) .*"""
)

st.sidebar.title("Control Panel")
# User inputs on the control panel
st.sidebar.subheader("Model selection")
model_name_webapp = st.sidebar.selectbox(
    "Choose the model",
    ["LSTM", "Autoencoder", "Baseline"],
    help="The forecast is model dependant.",
)

top = st.columns(1)[0]


dic_models = preload_models(con)
run_uuid, model = load_model(model_name_webapp, dic_models)
data = json.loads(open(f"/workdir/artifacts/0/{run_uuid}/artifacts/scaler.json", "r").read())
window_size, scaler_name, forecast_size, target, model_name = get_model_param(run_uuid, con)

scaler = select_scaler_cls(scaler_name, data)
model = select_model_cls(model_name, model, forecast_size, target)


inputs = load_data_from_db()
inputs = process_data(inputs, use_covariates=False)
inputs = inputs.iloc[-window_size:, :]
preds = inference(model, scaler, inputs)

inputs["Labels"] = "Inputs"
last_input_day = inputs.index[-1]
preds["Labels"] = "Forecast"
first_forecast_day = preds.index[0]
inputs = inputs.append(preds)
inputs = inputs.reset_index()
inputs["nbr_travels"] = inputs["nbr_travels"].apply(lambda x: int(x))

volume_fig = (
    alt.Chart(inputs)
    .encode(alt.X("index:T", title="Days"))
    .mark_line(color="#ffbb78", size=4, point=True)
    .encode(
        y=alt.Y("nbr_travels", axis=alt.Axis(title="Number of trains running", titleColor="#ff7f0e")),
        color="Labels:O",
        tooltip=[alt.Tooltip("nbr_travels", title="Number of trains"), alt.Tooltip("index:T", title="Day")],
    )
)

red_area = pd.DataFrame(
    data=[
        [last_input_day, inputs["nbr_travels"].max()],
        [first_forecast_day, inputs["nbr_travels"].max()],
    ],
    columns=["date", "nbr_travels"],
)

worst_case_area = alt.Chart(red_area).mark_area(opacity=0.5, color="red").encode(x="date:T", y="nbr_travels:Q")


fig = alt.layer(volume_fig, worst_case_area).configure_axis(labelFontSize=12, titleFontSize=16)

if st.util.env_util.is_repl():
    fig.save("nbr_travels_data.svg")

top.subheader(f"Observed data and forecast using {model_name_webapp}")
top.altair_chart(fig, use_container_width=True)

st.markdown("*Model trained on 2021-08-31*")
left_col, middle_col, right_col = st.columns(3)


plot_train_loss(run_uuid, ["loss", "val_loss"], con, left_col, title="Loss")
plot_train_loss(run_uuid, ["mean_absolute_error", "val_mean_absolute_error"], con, middle_col, title="MAE")


# Write key outputs in the control panel
right_col.subheader("Performance over test set")

right_col.markdown("**Observed sessions:** ici")
right_col.markdown("**Observed click rate:** ")
right_col.markdown("**Mean posterior click rate:** ")
right_col.markdown("**80% credible region for click rate:** posterior.ppf(0.1):.4f")
right_col.markdown("**P(click rate < than critical threshold):** ")
right_col.subheader("***Final decision:***")
