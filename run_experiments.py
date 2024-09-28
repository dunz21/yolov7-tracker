from detect_and_track import detect
from reid.VideoData import VideoData
from reid.VideoOption import VideoOption
from reid.InferenceParams import InferenceParams
import torch
import requests
import os
from pipeline.main import process_pipeline_mini,process_save_bd_pipeline
from config.api import APIConfig
import time
import traceback
from fastapi import FastAPI, HTTPException
from typing import List
import os
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()
SOLIDER_WEIGHTS ='transformer_120.pth'
footage_root_folder_path =  '/home/diego/mydrive/experiments'
results_root_folder_path = '/home/diego/mydrive/experiments_results'
APIConfig.initialize('http://localhost:5001')

# Define Pydantic models for validation
class ZoneCoordinates(BaseModel):
    coordinates: List[List[List[int]]]

class Zone(BaseModel):
    id: int
    name: str
    zone_type_name: str
    coordinates: List[List[List[int]]]
    channel_id: Any

class InferenceParams(BaseModel):
    id: int
    name: str
    value_params: Dict[str, Any]

class VideoDataInput(BaseModel):
    video_path: str
    zone: Zone
    inference_params: InferenceParams

@app.get('/get_experiments_videos', response_model=List[str])
def get_experiments_videos():
    # This returns all the video files in the specified folder
    directory = '/home/diego/mydrive/experiments'
    video_extensions = ('.mp4', '.mkv')  # Add more extensions if needed
    try:
        # List all files in the directory with the given video extensions
        videos = [f for f in os.listdir(directory) if f.endswith(video_extensions)]
        if not videos:
            raise HTTPException(status_code=404, detail="No video files found.")
        return videos
    except Exception as e:
        # Raise a 500 error if there's any issue
        raise HTTPException(status_code=500, detail=str(e))



# The function to run experiments
def run_experiments(video_path: str, inference_params: InferenceParams, zone: Zone):
    # Initialize the video data object
    videoDataObj = VideoData()
    videoDataObj.setBaseFolder(footage_root_folder_path)
    videoDataObj.setZoneInOutArea(zone.coordinates)

    # Prepare the folder results path
    folder_results_path = os.path.join(results_root_folder_path, "client_id", "store_id", "camera_channel_id")
    
    # Prepare the inference parameters
    inferenceParams = {
        'inference_params_id': inference_params.id,
        'inference_params_name': inference_params.name,
        'weights_folder': inference_params.value_params.get('weights_folder', 'yolov7.pt'),
        'yolo_model_version': inference_params.value_params.get('yolo_model_version', 'yolov7'),
        'tracker': inference_params.value_params.get('tracker', 'sort'),
        'save_all_images': inference_params.value_params.get('save_all_images', False),
        'bbox_centroid': inference_params.value_params.get('bbox_centroid', None),
    }

    # Example of how the VideoOption object might be initialized
    videoOptionObj = VideoOption(
        folder_results=folder_results_path,
        inferenceParams=inferenceParams,
        debug_mode=True,  # DEBUG MODE
        show_config=True,  # DEBUG MODE
        keep_resulting_video=True,  # DEBUG MODE
        compress_video=False,  # DEBUG MODE
        view_img=False,  # DEBUG MODE
        wait_for_key=False,  # DEBUG MODE
    )

    # Run the experiment pipeline
    try:
        with torch.no_grad():
            videoPipeline = detect(videoDataObj, videoOptionObj)
            process_pipeline_mini(csv_box_name=videoPipeline.csv_box_name, img_folder_name=videoPipeline.img_folder_name, solider_weights=SOLIDER_WEIGHTS)
        return {"status": "success", "message": "Experiment completed successfully"}
    except Exception as e:
        print("Error in detect")
        print(f"Error: {e}")
        traceback.print_exc()  # This will print the full traceback, including the line number
        return {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }

# POST endpoint to trigger experiments
@app.post('/run_experiment')
def run_experiment(video_data_input: VideoDataInput):
    video_path = video_data_input.video_path
    zone = video_data_input.zone
    inference_params = video_data_input.inference_params

    # Validate that the video file exists
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file {video_path} not found")

    # Run the experiment
    result = run_experiments(video_path, inference_params, zone)

    return result
            
            
