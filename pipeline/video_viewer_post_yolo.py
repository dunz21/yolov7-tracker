import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import pymysql
from config.api import APIConfig
import os

def prepare_event_timestamps_data(db, queue_video_id, start_video_time):
    event_type = 'In'
    start_time = datetime.strptime(start_video_time, '%H:%M:%S')

    conn = sqlite3.connect(db)
    bbox = pd.read_sql('SELECT * FROM bbox_raw', conn)
    conn.close()

    filtered_df = bbox[bbox['direction'] == event_type]
    result = filtered_df.groupby('id').first().reset_index()
    result = result[['id', 'time_video']]
    result_list = result.to_dict(orient='records')

    for record in result_list:
        time_video = datetime.strptime(record['time_video'], '%H:%M:%S')
        adjusted_time_video = (start_time + timedelta(hours=time_video.hour, minutes=time_video.minute, seconds=time_video.second)).time()
        record['bbox_id'] = record['id']  # Ensure bbox_id is included
        record['event_type'] = event_type
        record['queue_video_id'] = queue_video_id
        record['adjusted_time_video'] = adjusted_time_video.strftime('%H:%M:%S')  # Convert time object to string
        record['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime object to string
        record['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime object to string

    return result_list
    
def save_event_timestamps_to_api(event_timestamps=[]):
    try:
        APIConfig.save_event_timestamps(event_timestamps)
    finally:
        print("Event timestamps processed")



if __name__ == '__main__':    
    db_path = '/home/diego/mydrive/results/1/10/8/apumanque_entrada_2_20240720_1000/apumanque_entrada_2_20240720_1000_bbox.db'
    queue_video_id = 72
    start_video_time = '09:00:00'
    
    base_url_api = os.getenv('BASE_URL_API', 'http://localhost:1001')
    APIConfig.initialize(base_url_api)
    data = prepare_event_timestamps_data(db_path, queue_video_id, start_video_time)
    save_event_timestamps_to_api(data)
