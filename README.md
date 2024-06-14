# yolov7-object-tracking

### Manual
```
# If there are any changes
docker build -t yolov7-tracker . 
docker run --gpus all --env-file .env --network host -v /home/diego/Documents/Footage/calper:/app/videos -v $(pwd)/runs/detect:/app/runs/detect yolov7-tracker
```