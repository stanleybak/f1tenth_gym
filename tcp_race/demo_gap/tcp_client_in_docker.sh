#!/bin/bash
# run controller code in docker. Arguments: <server_port> <driver_id> <gain>

# if you get a permission error when trying to remove the container, try: sudo aa-remove-unknown

if [ "$#" -eq 3 ]; then
    PORT=$1
    DRIVER_ID=$2
    GAIN=$3
else
    echo "expected 3 arguments: <server_port> <driver_id> <gain>"
    exit 1
	
fi

PREFIX=gap_gain${GAIN}

CONTAINER=${PREFIX}_driver${DRIVER_ID}_container
IMAGE=${PREFIX}_image

echo "Running in Docker using container name $CONTAINER and image name $IMAGE."

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

#docker build --build-arg CATEGORIES="$CATEGORIES" . -t $IMAGE
docker build . -t $IMAGE

# TODO: check if this one is needed
#docker run -d --name $CONTAINER $IMAGE tail -f /dev/null

docker run -a STDOUT -a STDERR --net="host" --name $CONTAINER $IMAGE python3 /race/client_tcp_race.py $PORT $DRIVER_ID $GAIN

#docker run -d --name $CONTAINER $IMAGE python3 /race/tcp_race.py $PORT $DRIVER_ID $GAIN 

# "docker ps" should now list the image as running

# to get a shell, remove the lines at the end that delete the container and do: "docker exec -it $CONTAINER bash"

#docker cp $CONTAINER:/${RESULT_FILE} ${RESULT_FILE}

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

echo "Done."
