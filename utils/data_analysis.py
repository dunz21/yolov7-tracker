import numpy as np
import torch
import datetime
import os
import base64
import pandas as pd
from collections import Counter
from tqdm import tqdm
import sqlite3
from pipeline.vit_pipeline import get_files
from utils.time import seconds_to_time
from utils.types import Direction

def get_direction_info(df):
    temp_df = df.copy()
    first_directions = temp_df.groupby('id')['direction'].first().reset_index()
    direction_counts = first_directions['direction'].value_counts().to_dict()
    result = {k.lower(): v for k, v in direction_counts.items()}
    return result
    
def get_overlap_undefined(df, offset_overlap, direction_type):
    assert 'time_sec' in df.columns, "DataFrame must include a 'time_sec' column."

    df_copy = df.dropna(subset=['img_name']).copy()
    # Filter rows where direction is In or Out and group by ID to find the min and max time_sec
    grouped = df_copy[df_copy['direction'].isin([Direction.In.value, Direction.Out.value])].groupby('id').agg(
        start=('time_sec', 'min'),
        direction=('direction', 'first'),
        end=('time_sec', 'max')
    ).reset_index()

    # Expand the time window by the offset_overlap
    grouped['start'] -= offset_overlap
    grouped['end'] += offset_overlap

    # Prepare the output dataframe
    result = []

    # Filter all undefined direction rows once for efficiency
    # undefined_rows = df_copy[df_copy['direction'] == 'undefined']
    undefined_rows = df_copy[df_copy['direction'].isin(direction_type)].groupby('id').agg(
        start=('time_sec', 'min'),
        direction=('direction', 'first'),
        end=('time_sec', 'max')
    ).reset_index()

    # Loop over each group and find overlaps with Undefined
    for _, row in grouped.iterrows():
        overlaps = undefined_rows[
            (undefined_rows['start'] >= row['start']) & (undefined_rows['end'] <= row['end']) |
            (undefined_rows['end'] >= row['start']) & (undefined_rows['end'] <= row['end']) |
            (undefined_rows['start'] >= row['start']) & (undefined_rows['start'] <= row['end']) |
            (row['start'] >= undefined_rows['start']) & (row['end'] <= undefined_rows['end'])
        ]
        for _, o_row in overlaps.iterrows():
            if row['id'] == o_row['id']:
                continue
            overlap_type = ''
            if o_row['start'] >= row['start'] and o_row['end'] <= row['end']:
                overlap_type = 'inside'
            elif o_row['end'] >= row['start'] and o_row['end'] <= row['end']:
                overlap_type = 'start_overlap'
            elif o_row['start'] >= row['start'] and o_row['start'] <= row['end']:
                overlap_type = 'end_overlap'
            elif row['start'] >= o_row['start'] and row['end'] <= o_row['end']:
                overlap_type = 'suprass'
            result.append({
                'id': row['id'],
                'direction': row['direction'],
                'start_time': pd.to_datetime(row['start'], unit='s').time(),
                'end_time': pd.to_datetime(row['end'], unit='s').time(),
                'id_overlap': o_row['id'],
                'direction_overlap': o_row['direction'],
                'overlap_type': overlap_type,
                'id_overlap_start_time': pd.to_datetime(o_row['start'], unit='s').time(),
                'id_overlap_end_time': pd.to_datetime(o_row['end'], unit='s').time(),
                'offset': offset_overlap,
                'count' : len([value for _,value in overlaps.iterrows() if value['id'] != row['id']])
            })

    # Convert result to DataFrame
    return pd.DataFrame(result)