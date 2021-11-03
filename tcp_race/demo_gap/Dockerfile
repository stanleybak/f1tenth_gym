# Dockerfile for tcp race test

FROM python:3.8-slim-buster

RUN echo "Starting..."

WORKDIR /race

COPY ./networking.py .
COPY ./gap_driver.py .
COPY ./client_tcp_race.py .
COPY ./requirements.txt .

# install other (required) dependencies
RUN pip3 install -r requirements.txt 
