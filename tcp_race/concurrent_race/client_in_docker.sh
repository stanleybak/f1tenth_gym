#!/bin/bash
# run custom controller code in docker. Arguments: <server_port> <driver_id> <name>

# if you get a permission error when trying to remove the container, try: sudo aa-remove-unknown

if [ "$#" -eq 4 ]; then
    PORT=$1
    DRIVER_ID=$2
    TARFILE=$3
    DOCKER_PREFIX=$4
else
    echo "expected 4 arguments: <server_port> <driver_id> <tarfile> <docker_prefix>. Got $#: $@"
    exit 1
fi

CONTAINER=${DOCKER_PREFIX}_id${DRIVER_ID}_container
IMAGE=${DOCKER_PREFIX}_image
DEBUGLOG=stdout_${DOCKER_PREFIX}.txt

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

{

echo "Running in Docker using container name $CONTAINER and image name $IMAGE."

docker build --build-arg TARFILE="$TARFILE" . -t $IMAGE

docker run -a STDOUT -a STDERR --net="host" --name $CONTAINER $IMAGE python3 /race/riders_client_tcp_race.py $PORT $DRIVER_ID "$DOCKER_PREFIX"
} 2>&1 | tee $DEBUGLOG

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

echo "Done."
