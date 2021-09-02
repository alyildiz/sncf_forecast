# the ouput shape of the model (meaning 5 days forecast)
FORECAST_SIZE = 5
# the input of the model (meaning 50 previous days)
WINDOW_SIZE = FORECAST_SIZE * 10
# the feature to forecast, the webapp only supports len(TARGET) == 1
# but models and metrics work with len(TARGET) == N
# target can be ["nbr_travels"], ["nbr_late_trains"], ["nbr_travels", "nbr_late_trains"]
TARGET = ["nbr_travels"]
# columns to scale with the scaler
COLUMNS_TO_SCALE = ["nbr_travels", "nbr_late_trains"]
NUM_FEATURES = len(TARGET)
MLRUNS_DIR = "/workdir/logs/mlruns"
# scaler used, scaler_name can be "standard" or "min_max"
SCALER_NAME = "standard"

MONTHS_TO_NUMBERS = {
    "janvier": "01",
    "février": "02",
    "mars": "03",
    "avril": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07",
    "août": "08",
    "septembre": "09",
    "octobre": "10",
    "novembre": "11",
    "décembre": "12",
}
