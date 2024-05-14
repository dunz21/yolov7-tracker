import numpy as np
import sqlite3
import pandas as pd
import pymysql
from utils.types import Direction
from utils.pipeline import get_files
from utils.time import seconds_to_time


def process_data(path,start_time):
    
    files = get_files(path)
    conn = sqlite3.connect(files['db'])
    conn.row_factory = sqlite3.Row
    direction_param = Direction.In.value
    
    db = conn
    list_visits = pd.read_sql_query("SELECT * FROM bbox_raw WHERE direction = ?;", db, params=(direction_param,))
    # list_visits = list_visits.dropna(subset=['img_name'])
    
    # Drop duplicates based on 'id' to keep only one row per id
    list_visits = list_visits.drop_duplicates(subset=['id'])
    
    # Add 'direction' column by splitting 'img_name' and extracting the fourth element
    # list_visits['direction'] = list_visits['img_name'].apply(lambda x: x.split('_')[3])
    
    # Filter dataframe for rows where direction is either 'In' or 'Out'
    # list_visits = list_visits[list_visits['direction'].isin([Direction.In.value, Direction.Out.value])]
    
    # Add 'time' column calculated from 'frame_number' divided by 15, rounded to hours
    hours, minutes, seconds = map(int, start_time.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    
    list_visits['time_calculated'] = list_visits['frame_number'].apply(lambda x: seconds_to_time((x / 15) + total_seconds))
    
    # Filter by direction based on input parameter
    list_visits = list_visits[list_visits['direction'] == direction_param]
    
    # Convert 'time_calculated' to datetime to facilitate grouping by hour
    list_visits['hour'] = pd.to_datetime(list_visits['time_calculated'],format='%H:%M:%S').dt.hour
    
    # Get the full hour range from min to max
    hours_range = range(list_visits['hour'].min(), list_visits['hour'].max() + 1)
    
    
    total_ids_between_18_19 = list_visits[(list_visits['hour'] >= 18) & (list_visits['hour'] < 19)][['id','frame_number','time_calculated']]
    #list_visits.to_csv('list_visits.csv')
    
    # Group data by hour and count the occurrences
    grouped_data = list_visits.groupby('hour').size().reindex(hours_range, fill_value=0).reset_index(name='count')
    grouped_data['time_calculated'] = grouped_data['hour'].apply(lambda x: f"{x:02}:00")
    grouped_data = grouped_data[['count', 'time_calculated']]
    
    return grouped_data.to_dict(orient='records')
    
def save_visits_to_mysql(list_visits_group_by_hour, store_id, date, connection):
    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO visits (`count`, `time`, `store_id`, `date`, `created_at`, `updated_at`) VALUES (%s, %s, %s, %s, NOW(), NOW())"
            for item in list_visits_group_by_hour:
                cursor.execute(sql, (item['count'], item['time_calculated'], store_id, date))
                print(f"Inserted {item['count']} visits at {item['time_calculated']}")
        connection.commit()
    finally:
        print(date)
        
        
def process_data_send_to_mysql(path,start_time,store_id,date,connection):
    list_visits_group_by_hour = process_data(path,start_time)
    save_visits_to_mysql(list_visits_group_by_hour, store_id, date, connection)
    

if __name__ == '__main__':
    path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_12_tobalaba_9mayo'
    start_time = '08:00:00'
    store_id = 3
    date = '2024-05-12'
    HOST, ADMIN, PASS, DB =  'mivo-db.cj2ucwgierrs.us-east-1.rds.amazonaws.com', 'admin', '58#64KDashz^bLrqTG2', 'mivo'
    connection = pymysql.connect(host=HOST, user=ADMIN, password=PASS, database=DB)
    
    
    # Only debug
    #process_data(path=path,start_time=start_time,date='2024-05-12',store_id=3)
    
    
    list_path = [
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_06_tobalaba_30abril',
        #     'start_time': '11:50:00',
        #     'date': '2024-04-30',
        # },
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_06_tobalaba_2mayo',
        #     'start_time': '10:06:00',
        #     'date': '2024-05-02',
        # },
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_3mayo',
        #     'start_time': '10:38:00',
        #     'date': '2024-05-03',
        # },
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_4mayo',
        #     'start_time': '10:23:00',
        #     'date': '2024-05-04',
        # },
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_5mayo',
        #     'start_time': '10:03:00',
        #     'date': '2024-05-05',
        # },
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_07_tobalaba_6mayo',
        #     'start_time': '10:27:00',
        #     'date': '2024-05-06',
        # },
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_13_tobalaba_7mayo',
        #     'start_time': '10:00:00',
        #     'date': '2024-05-07',
        # },
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_09_tobalaba_8mayo',
        #     'start_time': '10:31:00',
        #     'date': '2024-05-08',
        # },
        {
            'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_12_tobalaba_9mayo',
            'start_time': '10:00:00',
            'date': '2024-05-09',
        },
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_13_tobalaba_12mayo',
        #     'start_time': '10:00:00',
        #     'date': '2024-05-12',
        # },
        # {
        #     'path': '/home/diego/Documents/yolov7-tracker/runs/detect/2024_05_13_tobalaba_10mayo',
        #     'start_time': '10:00:00',
        #     'date': '2024-05-10',
        # },
    ]
    
    
    for item in list_path:
        process_data_send_to_mysql(item['path'],item['start_time'],store_id,item['date'],connection)
    