import json
import sqlite3

import streamlit as st
from src.models.select_model import select_model_cls
from src.scaler.select_scaler import select_scaler_cls
from utils import (display_metrics, get_end_time, get_model_param,
                   get_test_metric, get_timeline, load_model, plot_timeline,
                   plot_train_loss, preload_models, setup_page)

con = sqlite3.connect("/workdir/data/mlflow.db")
cursor = con.cursor()

model_name_webapp = setup_page()

top = st.columns(1)[0]

dic_models = preload_models(con)
run_uuid, model = load_model(model_name_webapp, dic_models)
data = json.loads(open(f"/workdir/artifacts/0/{run_uuid}/artifacts/scaler.json", "r").read())
window_size, scaler_name, forecast_size, target, model_name = get_model_param(run_uuid, con)
scaler = select_scaler_cls(scaler_name, data)

model = select_model_cls(model_name, model, forecast_size, target)

inputs, last_input_day, first_forecast_day = get_timeline(model, scaler, window_size)
plot_timeline(inputs, last_input_day, first_forecast_day, model_name_webapp, top)

end_time = get_end_time(run_uuid, con)

st.markdown(f"*Model trained on {end_time}*")

left_col, middle_col, right_col = st.columns(3)

plot_train_loss(run_uuid, ["loss", "val_loss"], con, left_col, title="Loss")
plot_train_loss(run_uuid, ["mean_absolute_error", "val_mean_absolute_error"], con, middle_col, title="MAE")

mape, mae, mse = get_test_metric(run_uuid, con)
display_metrics(run_uuid, mape, mae, mse, right_col)
