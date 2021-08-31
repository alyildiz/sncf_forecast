import os
import sys

from pymongo import MongoClient


def get_mongo_client():
    try:
        print("Connection to database.")

        client = MongoClient(
            host=os.environ["MONGODB_HOST"] + ":" + str(os.environ["MONGODB_PORT"]),
            serverSelectionTimeoutMS=1000,  # 1 second timeout
            username=os.environ["MONGO_INITDB_ROOT_USERNAME"],
            password=os.environ["MONGO_INITDB_ROOT_PASSWORD"],
        )
        # check is server is accessible
        db = client.admin
        db.command("serverStatus")

        print("Connection secured.")
        return client

    except Exception as e:
        print(f"Error : {e!r}")
        sys.exit(1)


def insert_data(db, data):
    for i, row in data.iterrows():
        dic = {
            "date": row["date"],
            "nbr_travels": row["nbr_travels"],
            "nbr_late_trains": row["nbr_late_trains"],
        }
        db.trains_statistics.insert_one(dic)
    print("Insertion done.")
