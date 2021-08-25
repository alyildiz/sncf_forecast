FROM mongo:4.4.5

USER root

RUN apt-get update
RUN apt-get install python3 python3-pip -y

WORKDIR /workdir

COPY ./setup.py /workdir/setup.py
COPY ./requirements.txt /workdir/requirements.txt
RUN pip3 install -r requirements.txt