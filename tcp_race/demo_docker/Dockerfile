# Dockerfile for tcp race test

FROM python:3.8-slim-buster

RUN echo "Starting..."

ARG NAME="Stony Brook University"

RUN echo "Starting $NAME..."

WORKDIR /race

ADD "./${NAME}-snapshot.tar.gz" /race/files

COPY ./networking.py .
COPY ./riders_client_tcp_race.py .

# install dependencies
RUN pip3 install numpy==1.17.4 pyyaml==5.3.1

WORKDIR /race/files/pkg
RUN pip3 install -r requirements.txt 
