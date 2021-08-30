import numpy as np
import pandas as pd
from src.constants import MONTHS_TO_NUMBERS
from src.utils import FrBusinessCalendar


def load_data(FILE_PATH):
    df = pd.read_excel(FILE_PATH)
    return df


def process_data(df, use_covariates):
    df = clean_data(df)
    df = add_missing_dates(df)

    holidays = FrBusinessCalendar().holidays(start=df.date.min(), end=df.date.max())
    if use_covariates:
        df = add_covariates(df, holidays=holidays)
    df["nbr_travels"] = df["nbr_travels"].astype(int)
    df["nbr_late_trains"] = df["nbr_late_trains"].astype(int)

    df.set_index("date", inplace=True)
    return df


def clean_data(df):
    df_tmp = df.copy()
    df_tmp["text_process"] = df_tmp["Text"].apply(lambda x: x.split(" ")[0])
    df_tmp = df_tmp.loc[df_tmp["text_process"] == "Le"]
    df_tmp = extract_text_column(df_tmp)
    # print('with fill')
    # df_tmp = manual_fill(df_tmp)
    df_tmp = convert_date_to_datetime(df_tmp)

    return df_tmp


def extract_text_column(df):
    dates, list_nbr_travels, list_nbr_late_trains = [], [], []
    for i, row in df.iterrows():
        data = row["Text"].split("\n")
        day, month, year = data[0].split(" ")[2:-1]
        nbr_travels = data[1].split(" : ")[-1]
        nbr_late_trains = data[2].split(" ")[-2]

        dates.append(day + "-" + MONTHS_TO_NUMBERS[month] + "-" + year)
        list_nbr_travels.append(nbr_travels)
        list_nbr_late_trains.append(nbr_late_trains)

    df["date"] = dates
    df["nbr_travels"] = list_nbr_travels
    df["nbr_late_trains"] = list_nbr_late_trains

    return df[["date", "nbr_travels", "nbr_late_trains"]]


def convert_date_to_datetime(df):
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    return df


def add_missing_dates(df):
    date_range = pd.date_range(start=df.date.min(), end=df.date.max()).difference(df.date)
    for missing_date in list(date_range):
        dic = {
            "date": missing_date,
            "nbr_travels": np.nan,
            "nbr_late_trains": np.nan,
        }
        df = df.append(dic, ignore_index=True)

    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    df = df.ffill(axis=0)

    return df


def add_covariates(data, holidays):
    df = data.copy()
    # df['is_weekend'] = ((pd.DatetimeIndex(df['date']).dayofweek) // 5 == 1).astype(int)
    # df['is_holiday'] = np.where(df['date'].isin(holidays), 1, 0)
    # df['is_working_day'] = np.where((df['is_weekend']==1) | (df['is_holiday']==1), 0, 1)
    df["day_of_month"] = df["date"].dt.strftime("%d").astype(int)
    # df['day_of_week'] = df['date'].dt.strftime("%w").astype(int)
    # df['day_of_year'] = df['date'].dt.strftime("%j").astype(int)
    df["month"] = df["date"].dt.strftime("%m").astype(int)
    # df['week'] = df['date'].dt.strftime("%W").astype(int)
    # df['year'] = df['date'].dt.strftime("%Y").astype(int)
    # df['year_month'] = df['date'].dt.strftime("%Y%m").astype(int)
    # df['quarter'] = df['date'].dt.quarter.astype(int)
    df["day_of_month_cos"] = df["day_of_month"].apply(lambda x: np.cos(np.pi * 2 * x / 31))
    df["day_of_month_sin"] = df["day_of_month"].apply(lambda x: np.sin(np.pi * 2 * x / 31))

    df["month_cos"] = df["month"].apply(lambda x: np.cos(np.pi * 2 * x / 12))
    df["month_sin"] = df["month"].apply(lambda x: np.sin(np.pi * 2 * x / 12))

    df["nbr_travels"] = df["nbr_travels"].astype(int)
    df["nbr_late_trains"] = df["nbr_late_trains"].astype(int)

    return df


def manual_fill(df):
    df_tmp = df.copy()
    dic = {
        "date": "10-03-2021",
        "nbr_travels": 17800,
        "nbr_late_trains": np.nan,
    }
    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "11-03-2021",
        "nbr_travels": 17900,
        "nbr_late_trains": np.nan,
    }
    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "12-03-2021",
        "nbr_travels": 18300,
        "nbr_late_trains": np.nan,
    }
    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "13-03-2021",
        "nbr_travels": 12500,
        "nbr_late_trains": np.nan,
    }
    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "14-03-2021",
        "nbr_travels": 11900,
        "nbr_late_trains": np.nan,
    }
    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "15-03-2021",
        "nbr_travels": 17800,
        "nbr_late_trains": np.nan,
    }
    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "16-03-2021",
        "nbr_travels": 17700,
        "nbr_late_trains": np.nan,
    }
    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "17-03-2021",
        "nbr_travels": 17500,
        "nbr_late_trains": np.nan,
    }
    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "18-03-2021",
        "nbr_travels": 17800,
        "nbr_late_trains": np.nan,
    }

    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "19-03-2021",
        "nbr_travels": 18250,
        "nbr_late_trains": np.nan,
    }

    df_tmp = df_tmp.append(dic, ignore_index=True)

    dic = {
        "date": "20-03-2021",
        "nbr_travels": 12800,
        "nbr_late_trains": np.nan,
    }

    df_tmp = df_tmp.append(dic, ignore_index=True)
    return df_tmp
