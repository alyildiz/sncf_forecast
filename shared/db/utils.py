import os
import sys

from pymongo import MongoClient


def get_mongo_client():
    try:
        print("Connection to client.")

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
    length = len(data["c"])
    for i in range(length):
        dic = {
            "timestamp": data["t"][i],
            "open": data["o"][i],
            "close": data["c"][i],
            "high": data["h"][i],
            "low": data["l"][i],
            "volume": data["v"][i],
        }
        db.price.insert_one(dic)
