from src.collect_data import get_historical_data

from shared.db.utils import get_mongo_client, insert_data


def main():
    mongo_client = get_mongo_client()
    db = mongo_client["BTC"]
    data = get_historical_data()
    insert_data(db=db, data=data)


if __name__ == "__main__":
    main()
