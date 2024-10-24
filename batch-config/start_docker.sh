#!/bin/bash

# Define container name
CONTAINER_NAME="rae-bringup"

# Remove any existing container with the same name
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Run the container
docker run -it \
    --name=${CONTAINER_NAME} \
    --restart=unless-stopped \
    -v /dev/:/dev/ \
    -v /sys/:/sys/ \
    --privileged \
    --net=host \
    luxonis/rae-ros-robot:humble \
    /bin/bash -c "ros2 launch rae_bringup robot.launch.py use_slam:=false"