# Dockerfile for tcp race test

#FROM python:3.8-slim-buster
FROM ros:noetic-robot-focal

RUN echo "Starting..."
    
RUN apt-get update && apt-get install -y python3-pip ros-noetic-ackermann-msgs

ARG TARFILE="dockerfile-needs-arg.tar.gz"

RUN echo "Starting with tarfile $TARFILE..."

WORKDIR /race

ADD "./${TARFILE}" /race/files

RUN mkdir /catkin_ws && mkdir /catkin_ws/src && cp -r /race/files/pkg /catkin_ws/src/pkg

COPY ./networking.py .
COPY ./riders_client_tcp_race.py .

# install dependencies
RUN pip3 install numpy==1.17.4 pyyaml==5.3.1

WORKDIR /race/files/pkg
RUN pip3 install -r requirements.txt 
