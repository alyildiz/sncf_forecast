from datetime import datetime, timedelta

import pandas as pd
from src.utils import get_finnhub_client


def get_historical_data():
    fc = get_finnhub_client()
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    now = now + timedelta(hours=2)
    before = now - timedelta(days=500)

    now = int(datetime.timestamp(now))
    before = int(datetime.timestamp(before))

    data = fc.crypto_candles("BINANCE:BTCUSDT", "D", before, now)

    assert data["s"] == "ok", "Can't acces the API."

    return data


def load_data_from_db(db, use_date: bool = True):
    btc_price = list(db.price.find({}))
    df = pd.DataFrame(btc_price)
    if use_date:
        df["date"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x))

    df["id"] = 1
    return df
