# yolov7-object-tracking

### Manual
```
# In the env set up to VIDEO_DIR_CONTAINER=/app/videos
# If there are any changes
docker build -t yolov7-tracker . 
docker run --gpus all --env-file .env --network host -v /home/diego/Documents/Footage/calper:/app/videos -v $(pwd)/runs/detect:/app/runs/detect yolov7-tracker
```


#### Manual and process more videos at one time

```
docker run --gpus all --env-file .env   -e VIDEO_FILE=tobalaba_2024-05-24.mp4   -e VIDEO_DATE=2024-05-24  -e name=tobalaba_2024-05-24   -v /home/diego/mydrive/:/app/videos   -v $(pwd)/runs/detect:/app/runs/detect  yolov7-tracker

```



### automatic_detect ###

```
#build paso 1
docker compose -f docker-compose-automatic-detect.yaml build
# usarlo paso 2
docker compose -f docker-compose-automatic-detect.yaml up


# De una 1 y 2
docker compose -f docker-compose-automatic-detect.yaml up --build 

```
```
# PASO 1 BUILD
docker compose -f docker-compose-automatic-detect.yaml build 

# PASO 2 Poder ejectur varias de estas instancias
docker run --rm \
  --env-file .env \
  -e FOOTAGE_ROOT_FOLDER_PATH=/home/diego/mydrive/footage \
  -e RESULTS_ROOT_FOLDER_PATH=/home/diego/mydrive/results \
  -e BASE_URL_API=http://localhost:5001 \
  --volume $(pwd)/runs/detect:/app/runs/detect \
  --volume /home/diego/mydrive/footage:/home/diego/mydrive/footage \
  --volume /home/diego/mydrive/results:/home/diego/mydrive/results \
  --network host \
  --gpus '"device=0"' \
  yolov7-tracker-automatic_detect_service \
  python3 automatic_detect.py

  ## Paso 2 Detach

  docker run -d --rm \
  --env-file .env \
  -e FOOTAGE_ROOT_FOLDER_PATH=/home/diego/mydrive/footage \
  -e RESULTS_ROOT_FOLDER_PATH=/home/diego/mydrive/results \
  -e BASE_URL_API=http://localhost:5001 \
  --volume $(pwd)/runs/detect:/app/runs/detect \
  --volume /home/diego/mydrive/footage:/home/diego/mydrive/footage \
  --volume /home/diego/mydrive/results:/home/diego/mydrive/results \
  --network host \
  --gpus '"device=0"' \
  yolov7-tracker-automatic_detect_service \
  python3 automatic_detect.py


```

### ECR AWS
```
docker compose -f docker-compose-automatic-detect.yaml build
docker tag mivo/yolov7-tracker:latest 182436672416.dkr.ecr.us-east-1.amazonaws.com/mivo/yolov7-tracker:latest
docker push 182436672416.dkr.ecr.us-east-1.amazonaws.com/mivo/yolov7-tracker:latest
```

```
# Obtain Pass
aws ecr get-login-password --region us-east-1

```