pkill -f "morphling_device"

# if docker contrianer rabbitmq is running, stop it
if [ "$(docker ps -q -f name=rabbitmq)" ]; then
    docker stop rabbitmq
fi


# start rabbitmq container, set MAX_MSG_SIZE to 128MB
docker run -dit --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:4.0-management
sleep 5


# if docker contrianer redis is running, stop it
if [ "$(docker ps -q -f name=redis)" ]; then
    docker stop redis
fi

docker run -dit --rm --name redis -p 6379:6379 redis
sleep 5