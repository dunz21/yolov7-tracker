import os
import sqlite3
import pandas as pd
import subprocess
import boto3
import pymysql
from utils.types import Direction
from reid.matches import extract_reid_matches
from utils.time import convert_time_to_seconds
from pipeline.mysql_config import get_connection
from config.api import APIConfig


def extract_short_visits(video_path='', db_path='', max_distance=0.4, min_time_diff='00:00:10', max_time_diff='00:01:00', direction_param='In', fps=15, limit_vistits=10):
    # Create the 'clips' directory if it doesn't exist
    clips_dir = os.path.join(os.path.dirname(video_path), 'clips')
    if not os.path.exists(clips_dir):
        os.makedirs(clips_dir)
        print(f"Created directory for clips: {clips_dir}")
    else:
        print(f"Using existing directory for clips: {clips_dir}")
    
    # Connect to the database
    print(f"Connecting to database at {db_path}...")
    
    list_visits = extract_reid_matches(db_path, max_distance, min_time_diff, max_time_diff, fps, limit=limit_vistits)
    
    print(f"Found {len(list_visits)} visits matching criteria.")

    clip_paths = []

    # Process each row in the result set and create clips
    for index, row in enumerate(list_visits.to_dict(orient='records')):
        start_in_seconds = convert_time_to_seconds(row['start_in'])
        start_out_seconds = convert_time_to_seconds(row['start_out'])
        duration = start_out_seconds - start_in_seconds + 3 # Add 3 seconds to the duration
        
        clip_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{row['id_out']}_{row['id_in']}.mp4"
        clip_path = os.path.join(clips_dir, clip_name)
        
        # ffmpeg command to extract the clip
        ffmpeg_command = [
            'ffmpeg',
            '-ss', row['start_in'],
            '-i', video_path,
            "-vf", f"scale=1280:720",
            "-c:v", "h264_nvenc",
            "-cq:v", '40',
            "-preset", 'slow',
            '-t', str(duration),
            "-c:a", "copy",  
            clip_path,
            '-y',  # Overwrite output files without asking
        ]
        
        print(f"Extracting clip {index+1}/{len(list_visits)}: '{clip_name}' with duration {duration} seconds")
        
        # Run the ffmpeg command
        subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"Clip saved to: {clip_path}")
        clip_paths.append(clip_path)
    print("Database connection closed.")
    print("All clips extracted successfully.")
    return clip_paths

def upload_to_s3(file_path, bucket_name, s3_path):
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, bucket_name, s3_path)
    print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_path}")

def save_short_visits_to_api(short_video_clips_urls=[], date='', store_id=''):
    try:
        APIConfig.save_short_visits(short_video_clips_urls, date, store_id)
    finally:
        print(f"Short visits saved for date {date}")

def _get_file_size_in_mb(file_path):
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def process_clips_to_s3(short_video_clips=[], client_id='', store_id='', date='', pre_url='', bucket_name='videos-mivo'):
    s3_base_path = f"clients/{client_id}/stores/{store_id}/{date}/"
    urls = []

    print(f"Starting to process {len(short_video_clips)} clips for client '{client_id}' at store '{store_id}' on date '{date}'")

    for idx, clip_path in enumerate(short_video_clips):
        clip_name = os.path.basename(clip_path)
        s3_path = f"{s3_base_path}{clip_name}"
        clip_size_mb = _get_file_size_in_mb(clip_path)
        
        print(f"Processing clip {idx+1}/{len(short_video_clips)}: '{clip_name}' ({clip_size_mb:.2f} MB)")
        
        upload_to_s3(clip_path, bucket_name, s3_path)
        
        url = f"{pre_url}/{s3_path}"
        urls.append({'url': url})

        print(f"Uploaded '{clip_name}' to S3 at '{s3_path}'")

    print("All clips processed and uploaded to S3.")
    return urls


def main(path, client_id, store_id, date, pre_url):
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    connection = pymysql.connect(host=HOST, user=ADMIN, password=PASS, database=DB)
    save_short_visits_to_api(clips_urls, date, store_id)
    connection.close()

if __name__ == '__main__':
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_09_tobalaba_8mayo'
    date = '2024-05-08'
    store_id = 3
    client_id = 1
    pre_url = 'https://d12y8bglvlc9ab.cloudfront.net'
    
    HOST, ADMIN, PASS, DB =  'mivo-db.cj2ucwgierrs.us-east-1.rds.amazonaws.com', 'admin', '58#64KDashz^bLrqTG2', 'mivo'
    connection = get_connection(HOST, ADMIN, PASS, DB)

    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_06_tobalaba_2mayo'
    date = '2024-05-02'
    
    main(path, client_id, store_id, date, pre_url)
    
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_api(clips_urls, date, store_id)
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_3mayo'
    date = '2024-05-03'
    
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_api(clips_urls, date, store_id)
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_4mayo'
    date = '2024-05-04'
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_api(clips_urls, date, store_id)
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_5mayo'
    date = '2024-05-05'
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_api(clips_urls, date, store_id)
    
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_6mayo'
    date = '2024-05-06'
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_api(clips_urls, date, store_id)
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_13_tobalaba_7mayo'
    date = '2024-05-07'
    
    
    
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_api(clips_urls, date, store_id)
    connection.close()
