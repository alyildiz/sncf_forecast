FORECAST_SIZE = 5
WINDOW_SIZE = FORECAST_SIZE * 10
TARGET = ["nbr_travels"]
COLUMNS_TO_SCALE = ["nbr_travels", "nbr_late_trains"]
NUM_FEATURES = len(TARGET)
MLRUNS_DIR = "/workdir/logs/mlruns"
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
