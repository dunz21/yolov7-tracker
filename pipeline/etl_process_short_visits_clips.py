import os
import sqlite3
import pandas as pd
import subprocess
import boto3
import pymysql
from utils.types import Direction
from pipeline.vit_pipeline import get_files
from utils.time import convert_time_to_seconds
from pipeline.mysql_config import get_connection

def extract_short_visits(video_path='',db_path='', max_distance=0.21, min_time_diff='00:01:00', max_time_diff='00:02:00', direction_param='In', fps=15):
    # Create the 'clips' directory if it doesn't exist
    clips_dir = os.path.join(os.path.dirname(video_path), 'clips')
    if not os.path.exists(clips_dir):
        os.makedirs(clips_dir)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Query the database with parameterized values
    query = """
        WITH bboxraw AS (
            SELECT r.id, strftime('%H:%M:%S', '2000-01-01 00:00:00', (r.frame_number / ?) || ' seconds') AS start
            FROM bbox_raw r 
            GROUP BY id 
        ),
        intermediate AS (
            SELECT 
                r.id_out,
                CAST(MAX(r.distance) AS REAL) AS max_distance,
                r.id_in,
                r.time_diff,
                br_in.start AS start_in,
                br_out.start AS start_out
            FROM reranking_matches rm
            JOIN reranking r ON rm.id_out = r.id_out AND rm.id_in = r.id_in
            JOIN bboxraw br_out ON br_out.id = r.id_out 
            JOIN bboxraw br_in ON br_in.id = r.id_in 
            GROUP BY r.id_out, r.id_in, r.time_diff, br_in.start, br_out.start
        )
        SELECT
            id_out,
            max_distance,
            id_in,
            time_diff,
            start_in,
            start_out
        FROM intermediate
        WHERE max_distance < ?
          AND time_diff >= ?
          AND time_diff <= ?
        ORDER BY time_diff, max_distance ASC;
    """
    
    params = (fps, max_distance, min_time_diff, max_time_diff)
    list_visits = pd.read_sql_query(query, conn, params=params)
    
    clip_paths = []

    # Process each row in the result set and create clips
    for index, row in list_visits.iterrows():
        start_in_seconds = convert_time_to_seconds(row['start_in'])
        start_out_seconds = convert_time_to_seconds(row['start_out'])
        duration = start_out_seconds - start_in_seconds
        
        clip_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{row['id_out']}_{row['id_in']}.mp4"
        clip_path = os.path.join(clips_dir, clip_name)
        
        # ffmpeg command to extract the clip
        ffmpeg_command = [
            'ffmpeg',
            '-ss', row['start_in'],
            '-i', video_path,
            '-t', str(duration),
            '-c', 'copy',
            clip_path,
            '-y',  # Overwrite output files without asking
        ]
        
        # Run the ffmpeg command
        subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        clip_paths.append(clip_path)
    
    conn.close()
    return clip_paths

def upload_to_s3(file_path, bucket_name, s3_path):
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, bucket_name, s3_path)
    print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_path}")

def save_short_visits_to_mysql(short_video_clips_urls=[], date='', store_id='', connection=''):
    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO short_visits (url, date, store_id, created_at, updated_at) VALUES (%s, %s, %s, NOW(), NOW())"
            for item in short_video_clips_urls:
                cursor.execute(sql, (item['url'], date, store_id))
        connection.commit()
    finally:
        print(f"Short visits saved for date {date}")

def process_clips_to_s3(short_video_clips=[], client_id='', store_id='', date='', pre_url='', bucket_name='videos-mivo'):
    s3_base_path = f"clients/{client_id}/stores/{store_id}/{date}/"
    urls = []

    for clip_path in short_video_clips:
        clip_name = os.path.basename(clip_path)
        s3_path = f"{s3_base_path}{clip_name}"
        upload_to_s3(clip_path, bucket_name, s3_path)
        url = f"{pre_url}/{s3_path}"
        urls.append({'url': url})

    return urls


def main(path, client_id, store_id, date, pre_url):
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    connection = pymysql.connect(host=HOST, user=ADMIN, password=PASS, database=DB)
    save_short_visits_to_mysql(clips_urls, date, store_id, connection, pre_url)
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
    save_short_visits_to_mysql(clips_urls, date, store_id, connection, pre_url)
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_3mayo'
    date = '2024-05-03'
    
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_mysql(clips_urls, date, store_id, connection, pre_url)
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_4mayo'
    date = '2024-05-04'
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_mysql(clips_urls, date, store_id, connection, pre_url)
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_5mayo'
    date = '2024-05-05'
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_mysql(clips_urls, date, store_id, connection, pre_url)
    
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_6mayo'
    date = '2024-05-06'
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_mysql(clips_urls, date, store_id, connection, pre_url)
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_13_tobalaba_7mayo'
    date = '2024-05-07'
    
    
    
    clips_urls = process_clips_to_s3(path, client_id, store_id, date, pre_url)
    save_short_visits_to_mysql(clips_urls, date, store_id, connection, pre_url)
    connection.close()