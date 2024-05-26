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

## 1.- Agrergar columnas a la tabla, para que funcione el img_selection y el predict
def prepare_data_img_selection(db_path='', origin_table='', k_folds=4, n_images=5, new_table_name='bbox_img_selection'):
    """
    Apply KFold logic to data in a SQLite table and save the results to a new table.

    Parameters:
    - db_path: Path to the SQLite database file.
    - origin_table: Name of the source table to read data from.
    - k_folds: Number of folds for KFold.
    - n_images: Number of images to select per fold.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the new table (bbox_img_selection) exists, and drop it if it does
    cursor.execute("DROP TABLE IF EXISTS bbox_img_selection")
    conn.commit()

    # Read data from the specified table
    df = pd.read_sql(f"SELECT * FROM {origin_table}", conn)

    # Apply the logic from set_folds function
    if 'img_name' not in df.columns:
        raise ValueError("img_name column doesn't exist in the dataset.")

    df['k_fold'] = np.nan
    df['label_img'] = np.nan
    df['model_label_img'] = np.nan
    df['model_label_conf'] = np.nan
    df['k_fold_selection'] = np.nan
    df['selected_image'] = np.nan
    df_filtered = df[df['img_name'] != ''].copy()
    df_filtered.sort_values(by=['id', 'frame_number'], inplace=True)

    for id_value in df_filtered['id'].unique():
        subset = df_filtered[(df_filtered['id'] == id_value) & (df_filtered['img_name'].notna())]

        if len(subset) < k_folds * n_images:
            df.loc[subset.index, 'k_fold'] = 0
        else:
            kf = KFold(n_splits=k_folds)
            for fold, (_, test_index) in enumerate(kf.split(subset)):
                selected_indices = np.random.choice(test_index, min(n_images, len(test_index)), replace=False)
                df.loc[subset.iloc[selected_indices].index, 'k_fold'] = fold
                df.loc[subset.iloc[selected_indices].index, 'label_img'] = 0

    # Create a new table and write the modified DataFrame to it
    df.to_sql("bbox_img_selection", conn, if_exists="replace", index=False)

    # Close the connection to the database
    conn.close()

## 2.- Predict Image Selection, previo a seleccionar el topK
def predict_img_selection(db_file_path='', model_weights_path='', export_csv=False, csv_dir=None):
    conn = sqlite3.connect(db_file_path)
    
    # Load data from the database
    data = pd.read_sql('SELECT * FROM bbox_img_selection', conn)
    features = ['area', 'centroid_x', 'centroid_y', 'frame_number', 'overlap', 'distance_to_center', 'conf_score']
    
    # Ensure the input data contains all the required features
    assert all(f in data.columns for f in features), "Some required columns for prediction are missing."
    
    # Load the model
    model = joblib.load(model_weights_path)
    
    # Perform predictions
    predictions = model.predict(data[features])
    predicted_confidences = model.predict_proba(data[features]).max(axis=1)
    
    # Update DataFrame directly
    data['model_label_img'] = predictions.astype(int)
    data['model_label_conf'] = predicted_confidences.round(2)
    
    # Write updated DataFrame back to the database
    data.to_sql('bbox_img_selection', conn, if_exists='replace', index=False)
    
    if export_csv:
        if csv_dir is None:
            raise ValueError("csv_dir must be specified if export_csv is True")
        # Export to CSV
        csv_path = os.path.join(csv_dir, "img_selection_predicted.csv")
        data.to_csv(csv_path, index=False)
        print(f"Predictions exported to CSV at: {csv_path}")
    
    conn.close()

## 3.- Crear el topK de las imagenes seleccionadas, y guardar esa info en la BD
def clean_img_folder_top_k(db_file_path='', base_folder_images='', dest_folder_results='', k_fold=5, threshold=0.9, export_csv=False, csv_dir=None):
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # Load the modified data from the database
    data = pd.read_sql('SELECT * FROM bbox_img_selection WHERE img_name IS NOT NULL', conn)
    data.sort_values(by=['id', 'frame_number'], inplace=True)
    
    for id_value in data['id'].unique():
        local_threshold = threshold
        while True:
            id_df = data[data['id'] == id_value]
            filtered_id_df = id_df[(id_df['model_label_conf'] > local_threshold) & (id_df['model_label_img'] == 2)].copy()
            
            ### Esto solo ocurre cuando el modelo piensa que todas las imagenes son malas
            if local_threshold <= 0.5:
                filtered_id_df = id_df[((id_df['model_label_img'] == 2) | (id_df['model_label_img'] == 1)) & (id_df['model_label_conf'] > local_threshold)].copy()
            
            if len(filtered_id_df) >= k_fold or local_threshold <= 0:
                break
            local_threshold -= 0.05
        
        if len(filtered_id_df) >= k_fold:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=k_fold)
            
            for fold_number, (_, test_index) in enumerate(kf.split(filtered_id_df), start=1):
                fold_df = filtered_id_df.iloc[test_index]
                selected_row = fold_df.sample(n=1)
                selected_index = selected_row.index[0]
                
                # Update the database with fold and selection information
                sql = """UPDATE bbox_img_selection SET k_fold_selection = ?, selected_image = ? WHERE id = ? and img_name = ?"""
                cursor.execute(sql, (int(fold_number), True, int(selected_row.iloc[0]['id']),selected_row.iloc[0]['img_name']))
                
                # # Copy the selected image
                id_img = selected_row.iloc[0]['img_name'].split('_')[1]
                source_path = os.path.join(base_folder_images, id_img ,selected_row.iloc[0]['img_name'])
                dest_path = source_path.replace(base_folder_images, dest_folder_results)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(source_path, dest_path)
    
    conn.commit()
    conn.close()