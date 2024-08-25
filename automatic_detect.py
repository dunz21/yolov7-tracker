from detect_and_track import detect
from reid.VideoData import VideoData
from reid.VideoOption import VideoOption
from reid.VideoPipeline import VideoPipeline
from reid.InferenceParams import InferenceParams
import torch
import requests
import os
from pipeline.main import process_complete_pipeline,process_pipeline_mini,process_save_bd_pipeline
from config.api import APIConfig
import time
from tqdm import tqdm

#PARA PROD
# API + COMPLETE PIPELINE + NO SAVE VIDEO

#PARA DEBUG
# API + MINI PIPELINE + SAVE VIDEO

if __name__ == '__main__':
    # Total sleep time in seconds
    total_time = 1 * 60 * 60 + 40 * 60  # 1 hour and 20 minutes

    # Create a progress bar that lasts for total_time seconds
    # for _ in tqdm(range(total_time), desc="Waiting", ncols=100):
    #     time.sleep(1)  # Sleep for 1 second at a time
    PRODUCTION_MODE = True
    NO_SAVE_VIDEO = True
    
    footage_root_folder_path = os.getenv('FOOTAGE_ROOT_FOLDER_PATH', '/home/diego/mydrive/footage')
    results_root_folder_path = os.getenv('RESULTS_ROOT_FOLDER_PATH', '/home/diego/mydrive/results')
    
    base_url_api = 'https://api-v1.mivo.cl' if PRODUCTION_MODE else os.getenv('BASE_URL_API', 'http://localhost:1001')
    SOLIDER_WEIGHTS ='transformer_120.pth'
    APIConfig.initialize(base_url_api)

    while True:
        nextVideoInQueue = APIConfig.queue_videos()
        if not nextVideoInQueue:
            print("No more videos in the queue. Exiting.")
            break
        
        if not nextVideoInQueue['inference_params_name']:
            print("No inference params name. Exiting.")
            break
        
            
        inferenceParams = InferenceParams(
            weights_folder=nextVideoInQueue['inference_params_values']['weights_folder'],
            yolo_model_version=nextVideoInQueue['inference_params_values']['yolo_model_version'],
            tracker=nextVideoInQueue['inference_params_values']['tracker'],
            save_all_images=nextVideoInQueue['inference_params_values']['save_all_images']
        )
        
        videoDataObj = VideoData()
        videoDataObj.setBaseFolder(footage_root_folder_path)
        videoDataObj.setClientStoreChannel(nextVideoInQueue['client_id'], nextVideoInQueue['store_id'], nextVideoInQueue['camera_channel_id'])
        videoDataObj.setZoneFilterArea(nextVideoInQueue['zone_filter_area'])
        videoDataObj.setZoneInOutArea(nextVideoInQueue['zone_in_out_area'])
        videoDataObj.setVideoSource(nextVideoInQueue['video_file_name'])
        videoDataObj.setVideoMetaInfo(nextVideoInQueue['video_file_name'].split('.')[0], nextVideoInQueue['video_date'], nextVideoInQueue['video_time'])

        folder_results_path = os.path.join(results_root_folder_path, str(videoDataObj.client_id), str(videoDataObj.store_id), str(videoDataObj.camera_channel_id))
        videoOptionObj = VideoOption(
            folder_results=folder_results_path,
            noSaveVideo=NO_SAVE_VIDEO,
            weights=inferenceParams.weights_folder,
            model_version=inferenceParams.yolo_model_version,
            view_img=False,
            save_all_images=inferenceParams.save_all_images, #Es util solo en el video de la puerta, donde se requiere guardar todas las imagenes
            tracker_selection=inferenceParams.tracker
            )
        
    
        if not os.path.exists(videoDataObj.source):
            print(f"Video file {videoDataObj.source} does not exist. Skipping.")
            APIConfig.update_video_status(nextVideoInQueue['id'], 'not_found')
            continue
        
        with torch.no_grad():
            try:
                APIConfig.update_video_status(nextVideoInQueue['id'], 'processing')
                videoPipeline = detect(videoDataObj, videoOptionObj)
                APIConfig.update_video_status(nextVideoInQueue['id'], 'finished')
                if PRODUCTION_MODE:
                    process_complete_pipeline(
                        csv_box_name=videoPipeline.csv_box_name,
                        img_folder_name=videoPipeline.folder_name,
                        video_path=videoDataObj.source,
                        client_id=videoDataObj.client_id,
                        store_id=videoDataObj.store_id,
                        video_date=videoDataObj.video_date,
                        start_time_video=videoDataObj.video_time,
                        frame_rate=videoDataObj.frame_rate_video,
                        solider_weights=SOLIDER_WEIGHTS,
                        zone_type_id=nextVideoInQueue['zone_type_id']
                    )
                else:
                    process_pipeline_mini(csv_box_name=videoPipeline.csv_box_name, img_folder_name=videoPipeline.folder_name,solider_weights=SOLIDER_WEIGHTS)
                    
                    
                    
                results_example = f"{{'video': '{nextVideoInQueue['video_file_name'].split('.')[0]}', 'date' : '{nextVideoInQueue['video_date']}', 'time' : '{nextVideoInQueue['video_time']}' }}"
                APIConfig.post_queue_video_result(nextVideoInQueue['id'], 'yolov7', results_example)
            except Exception as e:
                print(e)
                print("Error in detect")
                APIConfig.update_video_status(nextVideoInQueue['id'], 'failed')
            
            
            
