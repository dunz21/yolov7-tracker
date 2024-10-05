from detect_and_track import detect
from reid.VideoData import VideoData
from reid.VideoOption import VideoOption
from reid.VideoPipeline import VideoPipeline
from reid.InferenceParams import InferenceParams
import torch
import requests
import platform
import os
from pipeline.main import process_complete_pipeline,process_pipeline_mini,process_save_bd_pipeline
from pipeline.compress_results_upload import pipeline_compress_results_upload,delete_local_results_folder
from config.api import APIConfig
from utils.print_prod_or_dev_mode import print_mode
from utils.download_video import find_video_in_s3,download_video_from_s3,delete_local_file
import time
from utils.types import QueueVideoStatus
from tqdm import tqdm
from dotenv import load_dotenv
from distutils.util import strtobool
import traceback
from utils.download_artifacts import check_and_download_files_from_s3
from utils.vastai_utils import destroy_instance
from utils.notifications import send_slack_notification
#PARA PROD
# API + COMPLETE PIPELINE + NO SAVE VIDEO

#PARA DEBUG
# API + MINI PIPELINE + SAVE VIDEO

if __name__ == '__main__':
    load_dotenv()
    check_and_download_files_from_s3()
    PRODUCTION_MODE = strtobool(os.getenv('PRODUCTION_MODE', False))
    CHANNEL_ID_FILTER = os.getenv('CHANNEL_ID_FILTER', None)
    KEEP_RESULTING_VIDEO = strtobool(os.getenv('KEEP_RESULTING_VIDEO', False))
    CLOUD_MACHINE = strtobool(os.getenv('CLOUD_MACHINE', True))
    DEBUG_MODE = strtobool(os.getenv('DEBUG_MODE', False))
    footage_root_folder_path = os.getenv('FOOTAGE_ROOT_FOLDER_PATH', '/home/diego/mydrive/footage')
    results_root_folder_path = os.getenv('RESULTS_ROOT_FOLDER_PATH', '/home/diego/mydrive/results')
    results_bucket = os.getenv('RESULTS_BUCKET_NAME')
    footage_bucket = os.getenv('FOOTAGE_BUCKET_NAME')
    machine_name = os.getenv('MACHINE_NAME')
    
    print_mode(PRODUCTION_MODE)
    
    SOLIDER_WEIGHTS ='transformer_120.pth'
    APIConfig.initialize(os.getenv('BASE_URL_API'))

    while True:
        nextVideoInQueue = APIConfig.queue_videos(CHANNEL_ID_FILTER)
        if not nextVideoInQueue:
            print("No more videos in the queue. Exiting.")
            break
        
        if not nextVideoInQueue['inference_params_name']:
            print("No inference params name. Exiting.")
            break
            
        inferenceParams = InferenceParams(
            inference_params_id=nextVideoInQueue.get('inference_params_id'),
            inference_params_name=nextVideoInQueue.get('inference_params_name'),
            weights_folder=nextVideoInQueue.get('inference_params_values', {}).get('weights_folder', 'yolov7.pt'),
            yolo_model_version=nextVideoInQueue.get('inference_params_values', {}).get('yolo_model_version', 'yolov7'),
            tracker=nextVideoInQueue.get('inference_params_values', {}).get('tracker', 'sort'),
            save_all_images=nextVideoInQueue.get('inference_params_values', {}).get('save_all_images', False),
            bbox_centroid=nextVideoInQueue.get('inference_params_values', {}).get('bbox_centroid', None),
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
            inferenceParams=inferenceParams,
            debug_mode=DEBUG_MODE, #DEBUG MODE
            show_config=True, #DEBUG MODE
            keep_resulting_video=KEEP_RESULTING_VIDEO, #DEBUG MODE
            compress_video=False, #DEBUG MODE
            view_img=False, #DEBUG MODE
            wait_for_key=False, #DEBUG MODE
            )
        
    
        if not os.path.exists(videoDataObj.source):
            exists_video_s3, video_s3_path = find_video_in_s3(footage_bucket, f"{videoDataObj.client_id}/{videoDataObj.store_id}/{videoDataObj.camera_channel_id}/{nextVideoInQueue['video_file_name']}", videoDataObj.video_date.replace('-', ''))
            if exists_video_s3:
                APIConfig.update_video_status(nextVideoInQueue['id'], QueueVideoStatus.DOWNLOADING.value,machine_name=machine_name)
                download_video_from_s3(footage_bucket,videoDataObj.source,video_s3_path, progress_callback=lambda progress: APIConfig.update_video_process_status(nextVideoInQueue['id'], progress, QueueVideoStatus.DOWNLOADING.value))
            else:    
                APIConfig.update_video_status(nextVideoInQueue['id'], QueueVideoStatus.NOTFOUND.value,machine_name=machine_name)
                continue
        
        with torch.no_grad():
            try:    
                APIConfig.update_video_status(nextVideoInQueue['id'],QueueVideoStatus.YOLO.value ,machine_name=machine_name)
                videoPipeline = detect(videoDataObj, videoOptionObj, progress_callback=lambda progress: APIConfig.update_video_process_status(nextVideoInQueue['id'], progress, QueueVideoStatus.YOLO.value))
                APIConfig.update_video_status(nextVideoInQueue['id'], QueueVideoStatus.REID_FEATURES.value,machine_name=machine_name)
                time_reid_features, time_reranking, time_video_encoding, time_video_upload =  process_complete_pipeline(
                    csv_box_name=videoPipeline.csv_box_name,
                    img_folder_name=videoPipeline.img_folder_name,
                    video_path=videoDataObj.source,
                    client_id=videoDataObj.client_id,
                    store_id=videoDataObj.store_id,
                    video_date=videoDataObj.video_date,
                    start_time_video=videoDataObj.video_time,
                    frame_rate=videoDataObj.frame_rate_video,
                    solider_weights=SOLIDER_WEIGHTS,
                    camera_channel_id=videoDataObj.camera_channel_id,
                    zone_type_id=nextVideoInQueue['zone_type_id'],
                    progress_callback=lambda progress, status : APIConfig.update_video_process_status(nextVideoInQueue['id'], progress, status)
                )

                    
                if DEBUG_MODE:
                    continue
                
                # Compress CSV, DB, imgs
                correct_upload = pipeline_compress_results_upload(videoPipeline.base_results_folder, f"{videoDataObj.client_id}/{videoDataObj.store_id}/{videoDataObj.camera_channel_id}/{nextVideoInQueue['video_date']}", results_bucket)
                
                if CLOUD_MACHINE and correct_upload:
                    delete_local_file(videoDataObj.source)
                    delete_local_results_folder(videoPipeline.base_results_folder)
                
         
                
                timings = {
                    'time_yolo': videoPipeline.metadata['time_yolo'],
                    'time_reid_features': f"{time_reid_features:.2f}",
                    'time_reranking': f"{time_reranking:.2f}",
                    'time_video_encoding': f"{time_video_encoding:.2f}",
                    'time_video_upload': f"{time_video_upload:.2f}"
                }
                
                
                APIConfig.post_queue_video_result(
                    nextVideoInQueue['id'],
                    time_start=videoPipeline.metadata['time_start'],
                    time_end=videoPipeline.metadata['time_end'],
                    timings=timings,
                    total_frames=videoPipeline.metadata['total_frames'],
                    total_duration=videoPipeline.metadata['total_duration'],
                    fps=videoPipeline.metadata['fps'],
                )
                 
                APIConfig.update_video_status(nextVideoInQueue['id'], QueueVideoStatus.FINISHED.value,machine_name=machine_name)
                send_slack_notification(f"Video {nextVideoInQueue['video_file_name']} it took {timings['time_yolo']} seconds in YOLO, {time_reid_features} seconds in reid features, {time_reranking} seconds in reranking, {time_video_encoding} seconds in video encoding and {time_video_upload} seconds in video upload")
                finished_queue_videos_process = APIConfig.get_finished_queue_videos(machine_name)
                if finished_queue_videos_process and CLOUD_MACHINE:
                    destroy_instance()
            except Exception as e:
                print("Error in detect")
                print(f"Error: {e}")
                traceback.print_exc()  # This will print the full traceback, including the line number
                error_msg = {
                    'error_message': str(e),
                    'traceback': traceback.format_exc(),
                }
                APIConfig.post_queue_video_result(nextVideoInQueue['id'],error=error_msg)
                APIConfig.update_video_status(nextVideoInQueue['id'], 'failed')
                # In case of an error, delete the video file if it was downloaded, to make space for the next video
                if CLOUD_MACHINE:
                    delete_local_file(videoDataObj.source)
                    
                    
            
            
            
