FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install python3.6 python3-pip -y

WORKDIR /workdir

COPY ./setup.py /workdir/setup.py
COPY ./requirements.txt /workdir/requirements.txt
RUN pip3 install -r requirements.txt

ENV TZ=Europe/Paris

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt install git -y