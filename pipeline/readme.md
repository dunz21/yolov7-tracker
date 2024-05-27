## convert_csv_to_sqlite
## swith_id_fixer

## prepare_data_img_selection
## predict_img_selection
## clean_img_folder_top_k

## get_features_from_model
## complete_re_ranking


Todo esto va a guardado en la BD SQLite para posterior uso


```
docker build -t yolov7-tracker .
docker run --gpus all --env-file .env  -v /home/diego/mydrive/:/app/videos -v $(pwd)/runs/detect:/app/runs/detect yolov7-tracker

docker run --gpus all --env-file .env \
  -e VIDEO_FILE=tobalaba_2024-05-22.mp4 \
  -e VIDEO_DATE=2024-05-22 \
  -e name=tobalaba_2024-05-22 \
  -v /home/diego/mydrive/:/app/videos \
  -v $(pwd)/runs/detect:/app/runs/detect \
  yolov7-tracker

# -d --rm antes de yolov7-tracker para detach y remover

```