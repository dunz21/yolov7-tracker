import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import pymysql
from config.api import APIConfig
import os
from reid.matches import extract_reid_matches


EVENT_TYPE_IN = 'In'
EVENT_TYPE_SHORT_VISIT = 'short_visit'



def prepare_event_timestamps_data(db, date, start_video_time, store_id):
    start_time = datetime.strptime(start_video_time, '%H:%M:%S')

    conn = sqlite3.connect(db)
    bbox = pd.read_sql('SELECT * FROM bbox_raw', conn)
    conn.close()

    filtered_df = bbox[bbox['direction'] == EVENT_TYPE_IN]
    result = filtered_df.groupby('id').first().reset_index()
    result = result[['id', 'time_video']]
    result_list = result.to_dict(orient='records')

    for record in result_list:
        time_video = datetime.strptime(record['time_video'], '%H:%M:%S')
        adjusted_time_video = (start_time + timedelta(hours=time_video.hour, minutes=time_video.minute, seconds=time_video.second)).time()
        record['bbox_id'] = record['id']  # Ensure bbox_id is included
        record['event_type'] = EVENT_TYPE_IN
        record['date'] = date
        record['store_id'] = store_id
        record['time_diff'] = None
        record['adjusted_time_video'] = adjusted_time_video.strftime('%H:%M:%S')  # Convert time object to string
        record['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime object to string
        record['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime object to string


    short_list_visits = extract_reid_matches(db, limit=1000)
    for record in short_list_visits.to_dict(orient='records'):
        time_video = datetime.strptime(record['start_in'], '%H:%M:%S')
        adjusted_time_video = (start_time + timedelta(hours=time_video.hour, minutes=time_video.minute, seconds=time_video.second)).time()
        record['bbox_id'] = record['id_in']
        record['event_type'] = EVENT_TYPE_SHORT_VISIT
        record['date'] = date
        record['store_id'] = store_id
        record['time_diff'] = record['time_diff']
        record['time_video'] = record['start_in']
        record['adjusted_time_video'] = adjusted_time_video.strftime('%H:%M:%S')
        record['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime object to string
        record['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime object to string
        result_list.append(record)

    return result_list
    
def save_event_timestamps(db_path='', date='', start_video_time='', store_id=''):
    try:
        event_timestamps = prepare_event_timestamps_data(db_path, date, start_video_time,store_id)
        APIConfig.save_event_timestamps(event_timestamps)
    finally:
        print("Event timestamps processed")



# if __name__ == '__main__':    
#     db_path = '/home/diego/mydrive/results/1/10/8/apumanque_entrada_2_20240818_1000/apumanque_entrada_2_20240818_1000_bbox.db'
#     date = '2024-08-18'
#     start_video_time = '10:00:00'
#     store_id = 10
    
#     base_url_api = os.getenv('BASE_URL_API', 'http://localhost:1001')
#     APIConfig.initialize(base_url_api)
#     data = prepare_event_timestamps_data(db_path, date, start_video_time,store_id)
#     save_event_timestamps(data)
