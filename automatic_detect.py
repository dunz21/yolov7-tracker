from detect_and_track import detect
from reid.VideoData import VideoData
from reid.VideoOption import VideoOption
from reid.VideoPipeline import VideoPipeline
import torch
import requests
import os
from pipeline.main import process_complete_pipeline,process_pipeline_mini
from config.api import APIConfig

if __name__ == '__main__':
    footage_root_folder_path = os.getenv('FOOTAGE_ROOT_FOLDER_PATH', '/home/diego/mydrive/footage')
    results_root_folder_path = os.getenv('RESULTS_ROOT_FOLDER_PATH', '/home/diego/mydrive/results')
    weights_folder = '/home/diego/Documents/MivoRepos/yolov10/runs/train/yolo_persons.v5i.yolov9_yolov10m-pc/weights/yolo_persons.v5_f012_yolov10m-pc.pt'
    base_url_api = os.getenv('BASE_URL_API', 'http://localhost:1001')
    
    APIConfig.initialize(base_url_api)

    while True:
        nextVideoInQueue = APIConfig.queue_videos()
        
        if not nextVideoInQueue:
            print("No more videos in the queue. Exiting.")
            break
        
        videoDataObj = VideoData()
        videoDataObj.setBaseFolder(footage_root_folder_path)
        videoDataObj.setClientStoreChannel(nextVideoInQueue['client_id'], nextVideoInQueue['store_id'], nextVideoInQueue['camera_channel_id'])
        videoDataObj.setZoneFilterArea(nextVideoInQueue['zone_filter_area'])
        videoDataObj.setZoneInOutArea(nextVideoInQueue['zone_in_out_area'])
        videoDataObj.setVideoSource(nextVideoInQueue['video_file_name'])
        videoDataObj.setVideoMetaInfo(nextVideoInQueue['video_file_name'].split('.')[0], nextVideoInQueue['video_date'], nextVideoInQueue['video_time'])

        folder_results_path = os.path.join(results_root_folder_path, str(videoDataObj.client_id), str(videoDataObj.store_id), str(videoDataObj.camera_channel_id))
        videoOptionObj = VideoOption(folder_results=folder_results_path,noSaveVideo=True, weights=weights_folder)
        
    
        if not os.path.exists(videoDataObj.source):
            print(f"Video file {videoDataObj.source} does not exist. Skipping.")
            APIConfig.update_video_status(nextVideoInQueue['id'], 'not_found')
            continue
        
        with torch.no_grad():
            try:
                APIConfig.update_video_status(nextVideoInQueue['id'], 'processing')
                videoPipeline = detect(videoDataObj, videoOptionObj)
                APIConfig.update_video_status(nextVideoInQueue['id'], 'finished')
                # process_complete_pipeline(videoPipeline.csv_box_name, videoPipeline.save_path, videoPipeline.folder_name, videoDataObj.client_id, videoDataObj.store_id, videoDataObj.video_date, videoDataObj.video_time, videoDataObj.frame_rate_video)
                process_pipeline_mini(csv_box_name=videoPipeline.csv_box_name, img_folder_name=videoPipeline.folder_name)
                results_example = f"{{'video': '{nextVideoInQueue['video_file_name'].split('.')[0]}', 'date' : '{nextVideoInQueue['video_date']}', 'time' : '{nextVideoInQueue['video_time']}' }}"
                APIConfig.post_queue_video_result(nextVideoInQueue['id'], 'yolov7', results_example)
            except Exception as e:
                print(e)
                print("Error in detect")
                APIConfig.update_video_status(nextVideoInQueue['id'], 'failed')
            
            
            
