from detect_and_track import detect
from reid.VideoData import VideoData
from reid.VideoOption import VideoOption
from reid.VideoPipeline import VideoPipeline
import torch
import requests
import os
from pipeline.main import process_pipeline
from config.api import APIConfig

if __name__ == '__main__':
    footage_root_folder_path = os.getenv('FOOTAGE_ROOT_FOLDER_PATH', '/home/diego/mydrive/footage')
    results_root_folder_path = os.getenv('RESULTS_ROOT_FOLDER_PATH', '/home/diego/mydrive/results')
    base_url_api = os.getenv('BASE_URL_API', 'http://localhost:8000')
    
    # Initialize the APIConfig with the base URL
    base_url_api = 'http://localhost:8000'
    APIConfig.initialize(base_url_api)

    while True:
        nextVideoInQueue = APIConfig.queue_videos()
        
        if not nextVideoInQueue:
            print("No more videos in the queue. Exiting.")
            break

        videoDataObj = VideoData()
        videoDataObj.setBaseFolder(footage_root_folder_path)
        videoDataObj.setClientStoreChannel(nextVideoInQueue['client_id'], nextVideoInQueue['store_id'], nextVideoInQueue['channel_id'])
        videoDataObj.setPolygonArea(nextVideoInQueue['zone_in'], nextVideoInQueue['zone_out'], nextVideoInQueue['zone_area'])
        videoDataObj.setVideoSource(nextVideoInQueue['video_file_name'])
        videoDataObj.setVideoMetaInfo(nextVideoInQueue['video_file_name'].split('.')[0], nextVideoInQueue['video_date'], nextVideoInQueue['video_time'])

        folder_results_path = os.path.join(results_root_folder_path, str(videoDataObj.client_id), str(videoDataObj.store_id), str(videoDataObj.channel_id))
        videoOptionObj = VideoOption(folder_results=folder_results_path)
        
        with torch.no_grad():
            try:
                APIConfig.update_video_status(nextVideoInQueue['id'], 'processing')
                videoPipeline = detect(videoDataObj, videoOptionObj)
                APIConfig.update_video_status(nextVideoInQueue['id'], 'finished')
            except Exception as e:
                print(e)
                print("Error in detect")
                APIConfig.update_video_status(nextVideoInQueue['id'], 'failed')
            process_pipeline(videoPipeline.csv_box_name, videoPipeline.save_path, videoPipeline.folder_name, videoDataObj.client_id, videoDataObj.store_id, videoDataObj.video_date, videoDataObj.video_time, videoDataObj.frame_rate_video)
