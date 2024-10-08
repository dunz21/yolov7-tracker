import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
from config.api import APIConfig
from reid.matches import extract_reid_matches
import numpy as np

ZONE_TYPES = [
    {
        'id': 1,
        'name': 'in_out_area',
    },
    {
        'id': 2,
        'name': 'filter_area',
    },
    {
        'id': 3,
        'name': 'exterior',
    },
]

def _get_direction_counts(db_path):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        query = '''
        SELECT direction, COUNT(*) as count
        FROM (
            SELECT id, direction
            FROM bbox_raw
            GROUP BY id
        )
        GROUP BY direction
        '''
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        directions = {row[0]: row[1] for row in results}
        return directions
            
def get_sales_by_date(store_id, date):
    unique_sales = APIConfig.get_unique_sales(store_id, date)
    return unique_sales

def get_exterior_data(db_path=''):
    # Connect to the exterior database and get direction counts
    directions_exterior = _get_direction_counts(db_path)

    # Compute total values for the exterior directions
    total_values_exterior = (
        directions_exterior.get('Undefined', 0) +
        directions_exterior.get('In', 0) +
        directions_exterior.get('Out', 0) +
        directions_exterior.get('Cross', 0) * 2
    )
    return total_values_exterior


    
def save_or_update_sankey(db_path='' ,store_id=0, date='',zone_type_id=1):
    if zone_type_id == ZONE_TYPES[0]['id']:
        # Entrance
        reid_matches_df = extract_reid_matches(db_path, max_distance=0.4, min_time_diff='00:00:10', max_time_diff='01:00:00', fps=15)
        reid_matches_df['time_diff_minutes'] = pd.to_timedelta(reid_matches_df['time_diff']).dt.total_seconds() / 60
        less_than_1_min_percentage = 0 if np.isnan((reid_matches_df['time_diff_minutes'] < 1).mean()) else (reid_matches_df['time_diff_minutes'] < 1).mean()
        directions_entrance = _get_direction_counts(db_path)
        total_entrance = directions_entrance.get('Out', 0)
        total_short_visits = int(total_entrance * less_than_1_min_percentage)
        total_sales = get_sales_by_date(store_id, date)
        APIConfig.save_sankey_diagram(store_id, date, short_visit=total_short_visits, interior=total_entrance, pos=total_sales)
    elif zone_type_id == ZONE_TYPES[2]['id']:
        # Exterior
        total_exterior = get_exterior_data(db_path)
        APIConfig.save_sankey_diagram(store_id, date, exterior=total_exterior)
    
    
    





if __name__ == '__main__':
    base_url_api = os.getenv('BASE_URL_API', 'http://localhost:1001')
    base_url_api = 'https://api-v1.mivo.cl/'
    APIConfig.initialize(base_url_api)

    store_id = 10
    
    folders = [
        # 'apumanque_entrada_2_20240714_0900',
        # 'apumanque_entrada_2_20240715_1209',
        # 'apumanque_entrada_2_20240716_1100',
        # 'apumanque_entrada_2_20240717_1000',
        # 'apumanque_entrada_2_20240718_1000',
        # 'apumanque_entrada_2_20240719_1000',
        # 'apumanque_entrada_2_20240720_1000',
        # 'apumanque_entrada_2_20240721_1000',
        # 'apumanque_entrada_2_20240722_1000',
        # 'apumanque_entrada_2_20240723_1000',
        # 'apumanque_entrada_2_20240724_1000',
        # 'apumanque_entrada_2_20240725_1011',
        # 'apumanque_entrada_2_20240726_1000',
        # 'apumanque_entrada_2_20240727_1000',
        'apumanque_entrada_2_20240728_1000',
        'apumanque_entrada_2_20240729_1000',
        # Add more folders as needed
    ]
    
    for folder in folders:
        db_path = f'/home/diego/mydrive/results/1/10/8/{folder}/{folder}_bbox.db'
        short_visit = extract_reid_matches(db_path)
        if short_visit is not None:
            # Extract and reformat the date
            date_str = folder.split('_')[3]
            date_formatted = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
            APIConfig.save_sankey_diagram(store_id, date_formatted, 0, 0, 0, 0, 0, short_visit)
            print(f"Processed data for folder: {folder}, found visits: {short_visit}")
        else:
            print(f"Failed to process data for folder: {folder}")

        
