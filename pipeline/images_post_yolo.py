import os
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

# Generate images by the CSV file
def generate_img_by_bbox(csv_path='', video_path='', img_path='', skip_frames=3, show_progress=True):
    # Load CSV data
    df = pd.read_csv(csv_path)
    max_frame_number = df['frame_number'].max()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Create the base folder for generated images
    img_generated_path = os.path.join(img_path)
    os.makedirs(img_generated_path, exist_ok=True)
    
    current_frame = 0
    if show_progress:
        progress_bar = tqdm(total=max_frame_number, desc="Processing frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame > max_frame_number:
            break
        
        if current_frame % skip_frames == 0:
            if show_progress:
                progress_bar.update(skip_frames)
            
            # Process each row in the DataFrame for the current frame
            frame_data = df[df['frame_number'] == current_frame]
            
            for _, row in frame_data.iterrows():
                img_id = row['id']
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                
                # Extract the image using bounding box coordinates
                cropped_img = frame[y1:y2, x1:x2]
                
                # Create a directory for the current id if it doesn't exist
                id_folder = os.path.join(img_generated_path, str(img_id))
                os.makedirs(id_folder, exist_ok=True)
                
                # Save the image
                img_save_path = os.path.join(id_folder, row['img_name'])
                cv2.imwrite(img_save_path, cropped_img)
        
        current_frame += 1
    
    cap.release()
    if show_progress:
        progress_bar.close()
    print("Image extraction completed!")