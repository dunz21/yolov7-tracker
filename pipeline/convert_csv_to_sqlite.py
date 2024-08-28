import datetime
import sqlite3
import pandas as pd
import cv2
import numpy as np
from shapely.geometry import LineString, Point
from sklearn.model_selection import KFold
import os
from sklearn.model_selection import KFold
import shutil
import sqlite3
import joblib  # For saving and loading the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from reid.utils import point_side_of_line

def convert_csv_to_sqlite(csv_file_path, db_file_path, table_name='bbox_raw'):
    """
    Convert a CSV file to a SQLite table and return the data from the table.
    
    Parameters:
    - csv_file_path: The file path of the CSV to be converted.
    - db_file_path: The file path of the SQLite database.
    - table_name: The name of the table where the CSV data will be inserted. Defaults to 'bbox_data'.
    
    Returns:
    - A pandas DataFrame containing the data from the specified SQLite table.
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Create a connection to the SQLite database
    with sqlite3.connect(db_file_path) as conn:
        # Write the data to a SQLite table
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Fetch the newly inserted data to verify
        fetched_data = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    
    # Return the fetched data
    return fetched_data 