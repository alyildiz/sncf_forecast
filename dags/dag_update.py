import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "description": "Use of the DockerOperator",
    "depend_on_past": False,
    "start_date": datetime(2021, 5, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

BASE_DIR = "/home/baris/PROJECTS/btc_forecast/"

dic_env = {
    "API_KEY": os.environ["API_KEY"],
    "API_KEY_SECRET": os.environ["API_KEY_SECRET"],
    "ACCESS_TOKEN": os.environ["ACCESS_TOKEN"],
    "ACCESS_TOKEN_SECRET": os.environ["ACCESS_TOKEN_SECRET"],
    "MONGODB_HOST": "172.24.0.3",
    "MONGODB_PORT": os.environ["MONGODB_PORT"],
    "MONGO_INITDB_ROOT_USERNAME": os.environ["MONGO_INITDB_ROOT_USERNAME"],
    "MONGO_INITDB_ROOT_PASSWORD": os.environ["MONGO_INITDB_ROOT_PASSWORD"],
}

with DAG("daily_update_new", default_args=default_args, schedule_interval="5 * * * *", catchup=False) as dag:
    update_db = DockerOperator(
        task_id="task_____daily_update_dbmongo",
        image="btc_forecast_update",
        environment=dic_env,
        container_name="task_____daily_update_dbmongo",
        api_version="auto",
        auto_remove=True,
        command="python3 /workdir/update.py",
        docker_url="unix://var/run/docker.sock",
        working_dir="/workdir",
        mount_tmp_dir=False,
        mounts=[
            Mount(source=BASE_DIR + "shared", target="/workdir/shared", type="bind"),
            Mount(source=BASE_DIR + "backend/modeling/src", target="/workdir/src", type="bind"),
            Mount(source=BASE_DIR + "backend/update", target="/workdir", type="bind"),
        ],
        network_mode="host",
    )

    update_lstm = DockerOperator(
        task_id="task_____daily_update_lstm",
        image="btc_forecast_modeling",
        environment=dic_env,
        container_name="task_____daily_update_lstm",
        api_version="auto",
        auto_remove=True,
        command="python3 /workdir/bin/train_model.py -m lstm",
        docker_url="unix://var/run/docker.sock",
        working_dir="/workdir",
        mount_tmp_dir=False,
        mounts=[
            Mount(source=BASE_DIR + "backend/modeling/bin", target="/workdir/bin", type="bind"),
            Mount(source=BASE_DIR + "backend/modeling/src", target="/workdir/src", type="bind"),
            Mount(source=BASE_DIR + "shared", target="/workdir/shared", type="bind"),
            Mount(source=BASE_DIR + "mlflow/db", target="/workdir/data", type="bind"),
            Mount(source=BASE_DIR + "mlflow/artifacts", target="/workdir/artifacts", type="bind"),
        ],
        network_mode="host",
    )

    update_baseline = DockerOperator(
        task_id="task_____daily_update_baseline",
        image="btc_forecast_modeling",
        environment=dic_env,
        container_name="task_____daily_update_baseline",
        api_version="auto",
        auto_remove=True,
        command="python3 /workdir/bin/train_model.py -m baseline",
        docker_url="unix://var/run/docker.sock",
        working_dir="/workdir",
        mount_tmp_dir=False,
        mounts=[
            Mount(source=BASE_DIR + "backend/modeling/bin", target="/workdir/bin", type="bind"),
            Mount(source=BASE_DIR + "backend/modeling/src", target="/workdir/src", type="bind"),
            Mount(source=BASE_DIR + "shared", target="/workdir/shared", type="bind"),
            Mount(source=BASE_DIR + "mlflow/db", target="/workdir/data", type="bind"),
            Mount(source=BASE_DIR + "mlflow/artifacts", target="/workdir/artifacts", type="bind"),
        ],
        network_mode="host",
    )

    update_autoencoder = DockerOperator(
        task_id="task_____daily_update_autoencoder",
        image="btc_forecast_modeling",
        environment=dic_env,
        container_name="task_____daily_update_autoencoder",
        api_version="auto",
        auto_remove=True,
        command="python3 /workdir/bin/train_model.py -m autoencoder",
        docker_url="unix://var/run/docker.sock",
        working_dir="/workdir",
        mount_tmp_dir=False,
        mounts=[
            Mount(source=BASE_DIR + "backend/modeling/bin", target="/workdir/bin", type="bind"),
            Mount(source=BASE_DIR + "backend/modeling/src", target="/workdir/src", type="bind"),
            Mount(source=BASE_DIR + "shared", target="/workdir/shared", type="bind"),
            Mount(source=BASE_DIR + "mlflow/db", target="/workdir/data", type="bind"),
            Mount(source=BASE_DIR + "mlflow/artifacts", target="/workdir/artifacts", type="bind"),
        ],
        network_mode="host",
    )

    update_db >> update_lstm
    update_db >> update_baseline
    update_db >> update_autoencoder
