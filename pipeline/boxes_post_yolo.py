import os
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.draw_tools import draw_polygon_interested_area,draw_boxes_entrance_exit,draw_boxes

def preprocess_frame_data(df):
    frame_data_dict = {}
    for _, row in df.iterrows():
        frame_number = row['frame_number']
        if frame_number not in frame_data_dict:
            frame_data_dict[frame_number] = []
        frame_data_dict[frame_number].append(row)
    return frame_data_dict

def draw_boxes_on_video(csv_path='', video_path='', output_video_path='', show_progress=True):
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Preprocess frame data for O(1) access
    frame_data_dict = preprocess_frame_data(df)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 12)
    fps = 15  # Set the desired fps
    
    # Get the duration of the video in seconds
    duration_in_seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the total number of frames
    total_frames = int(duration_in_seconds * fps)
    
    # Define the codec based on your choice
    codec = 'mp4v'  # Change this to the codec you want to use, e.g., 'h264', 'h265'
    print(f"Codec: {codec}")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    current_frame = 0
    if show_progress:
        progress_bar = tqdm(total=total_frames, desc="Processing frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # for _ in range(3):
        #     cap.grab()  # Skip reading frames
        
        if show_progress:
            progress_bar.update(1)
            
        # Process each row in the dictionary for the current frame
        if current_frame in frame_data_dict:
            frame_data = frame_data_dict[current_frame]
            bbox = []
            extra_info = {}
            
            for row in frame_data:
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                bbox.append((x1, y1, x2, y2, row['id'], row['conf_score']))
                extra_info[row['id']] = {'overlap': 0, 'distance': 0}
                extra_info[row['id']]['distance'] = row['distance_to_center']
                extra_info[row['id']]['overlap'] = row['overlap']
            
            # Draw bounding boxes
            if isinstance(frame, np.ndarray):
                draw_boxes(frame, bbox, extra_info=extra_info, color=(0, 0, 255), position='Top')
                draw_boxes_entrance_exit(frame)
                draw_polygon_interested_area(frame)
            else:
                print(f"Frame at {current_frame} is not a valid numpy array.")
        
        # Write the frame with drawn boxes to the output video
        if current_frame > 9000: 
            out.write(frame)
        
        current_frame += 1
    
    cap.release()
    out.release()
    if show_progress:
        progress_bar.close()
    print("Video processing completed!")
    

if __name__ == '__main__':
    csv = '/home/diego/Documents/yolov7-tracker/runs/detect/tobalaba_entrada_20240606_0900_PERFORMANCE_TEST/tobalaba_entrada_20240606_0900_PERFORMANCE_TEST_bbox.csv'
    video = '/home/diego/mydrive/footage/1/3/1/tobalaba_entrada_20240606_0900_PERFORMANCE_TEST.mkv'
    draw_boxes_on_video(csv_path=csv, video_path=video, output_video_path='/home/diego/Documents/yolov7-tracker/runs/detect/tobalaba_entrada_20240606_0900_PERFORMANCE_TEST/tobalaba_entrada_20240606_0900_PERFORMANCE_TEST_bbox2.mkv', show_progress=True)
    
