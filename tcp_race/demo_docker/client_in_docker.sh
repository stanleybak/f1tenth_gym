#!/bin/bash
# run custom controller code in docker. Arguments: <server_port> <driver_id> <name>

# if you get a permission error when trying to remove the container, try: sudo aa-remove-unknown

if [ "$#" -eq 4 ]; then
    PORT=$1
    DRIVER_ID=$2
    TAR_NAME=$3
    DOCKER_PREFIX=$4
else
    echo "expected 3 arguments: <server_port> <driver_id> <tar_name> <docker_prefix>"
    exit 1
fi

CONTAINER=${DOCKER_PREFIX}_id${DRIVER_ID}_container
IMAGE=${DOCKER_PREFIX}_image

echo "Running in Docker using container name $CONTAINER and image name $IMAGE."

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

docker build --build-arg NAME="$TAR_NAME" . -t $IMAGE

docker run -a STDOUT -a STDERR --net="host" --name $CONTAINER $IMAGE python3 /race/riders_client_tcp_race.py $PORT $DRIVER_ID "$TAR_NAME"

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

echo "Done."
