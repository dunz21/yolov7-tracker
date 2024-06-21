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
docker-compose -f docker-compose-automatic-detect.yaml build
# usarlo paso 2
docker-compose -f docker-compose-automatic-detect.yaml up


# De una 1 y 2
docker-compose -f docker-compose-automatic-detect.yaml up --build 

```