from src.extract_data import clean_data, load_data

from shared.db.utils import get_mongo_client, insert_data


def main():
    mongo_client = get_mongo_client()
    db = mongo_client["sncf"]

    df = load_data("/docker-entrypoint-initdb.d/sncfisajoke_user_tweets_31_08.xlsx")
    df = clean_data(df)

    insert_data(db=db, data=df)


if __name__ == "__main__":
    main()
