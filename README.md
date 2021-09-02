Production ready project : Daily running trains forecast
==============================

An Airflow DAG to collect the latest data available from https://twitter.com/sncfisajoke (this account is a bot that daily posts statistics about the status of the french railway company called SNCF ) and apply deep learning methods to forecast the number of running trains for the next N days.

There are currently 3 models implemented : 2 models are based on the tensorflow framework and the 3rd one is a baseline model. 
The current output of the baseline model is the average value for each day from its input.

![alt text](https://github.com/alyildiz/sncf_forecast/blob/master/web_app/webapp.jpg?raw=true)

About the code 
-----------
Each run of ```./backend/modeling/bin/train_model.py``` is saved in the mlflow database.

The web app loads the best LSTM and Autoencoder (and baseline) models that were found right after updating the data in the night. 

```docker-compose up``` to run all containers including Airflow.

- The mlflow webserver runs on ```localhost:5000```
- The web_app runs on ```localhost:8501```
- The airflow webserver runs on ```localhost:8080```
- The jupyter server from the modeling container runs on ```localhost:8888```


```docker exec -it <container_name> bash``` to get inside of a container.

Example :

```
docker exec -it modeling bash
python3 /workdir/bin/train_model.py -m lstm
```



Environnement Files 
-----------
```./.env``` : 

```
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=password
MONGODB_PORT=27017
AIRFLOW_UID=1000
AIRFLOW_GID=0
```

```./backend/db/.env``` :

```
MONGODB_HOST=localhost
```

```./backend/modeling/.env``` :

```
MONGODB_HOST=dbmongo
```

```./backend/update/.env``` :

```
MONGODB_HOST=dbmongo
API_KEY=twitter_api_key
API_KEY_SECRET=twitter_api_key_secret
ACCESS_TOKEN=twitter_access_token
ACCESS_TOKEN_SECRET=twitter_access_token_secret
```

```./web_app/.env``` :

```
MONGODB_HOST=dbmongo
```


Project Organization 
-----------


------------

    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    │
    │
    ├── backend                <- Source code for use in this project.
    │   ├── db                 <- Container to run the mongodb database that stores the data from the twitter account
    │   │
    │   ├── modeling           <- Modeling container that contains the ML pipeline
    │   │   ├── bin            <- Contains the training script 
    │   │   └── src            <- Contains the data loader, scalers, models, ... 
    │   │
    │   └── update             <- Update container that contains the update script
    │
    ├── dags                   <- Contains the DAG for Airflow
    │
    ├── mlflow                 <- Mlflow container that contains the mlflow sqlite database
    │
    ├── web_app                <- Web app container 
    │
    └── tox.ini                <- tox file with settings for running tox; see tox.readthedocs.io


--------
