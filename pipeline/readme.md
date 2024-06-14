#### Ojo estos son los pasos principales
- Procesar video con yolo, esto genera la carpeta de imaganes, el video resultante que termina pesando mucho, y el CSV que lo primero que se hace es pasarlo a sqlite
- El proceso se puede divir en 3 partes
  - Procesar video con yolo dando resultado a csv,imgs,video
  - Comprimir video
  - Pipeline completo 

Todo lo anterior se puede separar




# Pipieline 
## convert_csv_to_sqlite
## swith_id_fixer (Ocupa solider model)

## prepare_data_img_selection
## predict_img_selection (Aca se ocupa un modelo que lo mejor es que sea entrenado con cada data)
## clean_img_folder_top_k

## get_features_from_model
## complete_re_ranking

## etl_process_visits_per_time (Requiere credenciales MySQL)
## etl_process_short_visits_clips (Requeire S3)


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


### Cosas que he aprendido

- Esta el tema del docker-compose que al hacer build, toma los env en cuenta que hay hasta el momento, pero no puedo ocupar env variables dentro del mismo. Tuve el problema con querer poner dinamico el tema del volumen
- Al final saque el tema de docker compose y voy a hacer docker run no mas...
- Otra cosa que aprendi es que al abrir un terminal en VS Code el .env del workspace directory se carga... y cada vez que abro una nueva terminal se carga nuevamente con los datos.