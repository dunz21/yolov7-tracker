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
    # Initialize the APIConfig with the base URL
    APIConfig.initialize("http://localhost:8000")

    videoOptionObj = VideoOption()
    nextVideoInQueue = APIConfig.queue_videos()
    
    if nextVideoInQueue:
        base_folder_path = '/home/diego/mydrive/footage/'
        
        videoDataObj = VideoData()
        videoDataObj.setBaseFolder(base_folder_path)
        videoDataObj.setClientStoreChannel(nextVideoInQueue['client_id'], nextVideoInQueue['store_id'], nextVideoInQueue['channel_id'])
        videoDataObj.setPolygonArea(nextVideoInQueue['zone_in'], nextVideoInQueue['zone_out'], nextVideoInQueue['zone_area'])
        videoDataObj.setVideoSource(nextVideoInQueue['video_file_name'])
        videoDataObj.setVideoMetaInfo(nextVideoInQueue['video_file_name'].split('.')[0], nextVideoInQueue['video_date'], nextVideoInQueue['video_time'])
        videoDataObj.setDB('localhost', 'admin', 'root', 'mivo')
        
        with torch.no_grad():
            try:
                APIConfig.update_video_status(nextVideoInQueue['id'], 'processing')
                videoPipeline = detect(videoDataObj, videoOptionObj)
                APIConfig.update_video_status(nextVideoInQueue['id'], 'finished')
            except Exception as e:
                print(e)
                print("Error in detect")
                APIConfig.update_video_status(nextVideoInQueue['id'], 'failed')
                #continue
            process_pipeline(videoPipeline.csv_box_name, videoPipeline.save_path, videoPipeline.folder_name)
