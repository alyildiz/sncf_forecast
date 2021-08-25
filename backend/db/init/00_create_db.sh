#/bin/bash
echo ${PWD}

python3 /docker-entrypoint-initdb.d/create_db.py
#python3 /docker-entrypoint-initdb.d/fill_db.py