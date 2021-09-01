import os
from pprint import pprint as pp

import pandas as pd
import tweepy
from src.extract_data import clean_data, get_mongo_client

from shared.db.utils import insert_data


def main():
    auth = tweepy.OAuthHandler(os.environ["API_KEY"], os.environ["API_KEY_SECRET"])
    auth.set_access_token(os.environ["ACCESS_TOKEN"], os.environ["ACCESS_TOKEN_SECRET"])
    api = tweepy.API(auth)
    pp(os.environ)
    client = get_mongo_client()
    db = client["sncf"]

    userID = "sncfisajoke"

    df_db = pd.DataFrame(list(db.trains_statistics.find()))
    df_db = df_db.drop(["_id"], axis=1)
    df_db["date"] = pd.to_datetime(df_db["date"], format="%d-%m-%Y")
    df_db = df_db.sort_values(by=["date"], ascending=[True])
    latest_date_db = df_db["date"].values[-1]

    tweets = api.user_timeline(screen_name=userID, count=15, include_rts=False, tweet_mode="extended")

    df = pd.DataFrame([info.full_text for info in tweets], columns=["Text"])
    df = clean_data(df)
    df["date_datetime"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df = df.loc[df["date_datetime"] > latest_date_db].drop(["date_datetime"], axis=1)

    if df.shape[0] > 0:
        insert_data(db, df)
        print("Database updated.")
    else:
        print("No new data found.")


if __name__ == "__main__":
    main()
