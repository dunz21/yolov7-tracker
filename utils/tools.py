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
# Sirve para preparar el dataset para entrenar el modelo con el img. selection del labeler
# Tb va a tener anotada la prediccion del modelo


# Example usage
# db_path = "/home/diego/Documents/yolov7-tracker/runs/detect/bytetrack_santos_dumont/santos_dumont_bbox.db"
# origin_table = "bbox_raw"
# k_folds = 4
# n_images = 5
# prepare_data_img_selection(db_path, origin_table, k_folds, n_images)



## 0.- Convertir CSV BBOX a SQLite
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
    cursor = conn.cursor()
    
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
    
    # Update the database
    # TODO: FIX ACA ESTA haciendo update a muchos IDS
    for i, row in data.iterrows():
        sql = """UPDATE bbox_img_selection SET model_label_img = ?, model_label_conf = ? WHERE id = ? and img_name = ?"""
        cursor.execute(sql, (int(predictions[i]), round(predicted_confidences[i], 2), row['id'], row['img_name']))
    
    conn.commit()
    
    if export_csv:
        if csv_dir is None:
            raise ValueError("csv_dir must be specified if export_csv is True")
        # Export to CSV
        data['model_label_img'] = predictions
        data['model_label_conf'] = predicted_confidences.round(2)
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

### En construccion
# def train_and_save_model(training_data_path, model_save_path):
#     INTEREST_LABEL = 'label_img'
#     interest_values = [1, 2]

#     only_train_data = pd.read_csv(training_data_path)
#     only_train_data = only_train_data.dropna(subset=['img_name'])
#     only_train_data = only_train_data[only_train_data[INTEREST_LABEL].isin(interest_values)]

#     features = ['area', 'centroid_x', 'centroid_y', 'frame_number', 'overlap', 'distance_to_center', 'conf_score']
#     target = INTEREST_LABEL

#     assert all(f in only_train_data.columns for f in features + [target]), "Some required columns are missing."

#     X_train, X_val, y_train, y_val = train_test_split(only_train_data[features], only_train_data[target], test_size=0.2, random_state=42)

#     model = GradientBoostingClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     # Save the trained model
#     joblib.dump(model, model_save_path)

#     val_predictions = model.predict(X_val)
#     val_accuracy = accuracy_score(y_val, val_predictions)
#     print(f"Validation Accuracy: {val_accuracy}")


    
#     if export_csv:
#         if csv_dir is None:
#             raise ValueError("csv_dir must be specified if export_csv is True")
#         # Export to CSV
#         csv_path = os.path.join(csv_dir, "cleaned_images_selection.csv")
#         data.to_csv(csv_path, index=False)
#         print(f"Cleaned data exported to CSV at: {csv_path}")
#     conn.close()

### OTROS ####

def draw_boxes(img, bbox , offset=(0, 0),extra_info=None,color=None,position='Top'):
    for box in bbox:
        x1, y1, x2, y2,id,score = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        id = int(id)
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # id = int(identities[i]) if identities is not None else 0

        label = str(id) + ":" + "person"
        if extra_info is not None:
            label += str(f"oc:{extra_info[id]['overlap']:.2f}")
            label += str(f"di:{extra_info[id]['distance']:.2f}")

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if color is None:
            color = (255, 0, 20)
            # color_rect_text = (255, 144, 30)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if position == 'Top':
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        else:
            cv2.rectangle(img, (x1, y2 - 20), (x1 + w, y2), color, -1)
            cv2.putText(img, label, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

    return img

def calculate_overlap(rect1, rect2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both rectangles
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate union area
    union_area = area1 + area2 - intersection_area

    # Calculate the overlap percentage
    overlap = intersection_area / union_area

    return overlap

def distance_to_bbox_bottom_line(line=[], bbox=[]):
    """
    Calculate the distance between the closest point on a line and the center of the bottom edge of a bounding box (bbox).
    
    :param line: A list of points [[x1, y1], [x2, y2]] defining the line.
    :param bbox: A tuple representing the bounding box (x1, y1, x2, y2).
    :return: The shortest distance between the line and the center of the bottom edge of the bbox.
    """
    line = LineString(line)
    x1, y1, x2, y2 = bbox

    # Calculate the center of the bottom edge of the bbox
    bottom_center = Point((x1 + x2) / 2, y2)

    # Calculate the shortest distance from the bottom center to the line
    distance = bottom_center.distance(line)
    return distance

def seconds_to_time(seconds):
    # Create a timedelta object
    td = datetime.timedelta(seconds=seconds)
    # Add the timedelta to a minimal datetime object
    time = (datetime.datetime.min + td).time()
    # Convert to a string format
    return time.strftime("%H:%M:%S")

def number_to_letters(num):
    mapping = {i: chr(122 - i) for i in range(10)}
    num_str = str(num)
    letter_code = ''.join(mapping[int(digit)] for digit in num_str)
    return letter_code