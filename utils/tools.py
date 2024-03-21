
import sqlite3
import pandas as pd
import cv2
import numpy as np
from shapely.geometry import LineString, Point
from sklearn.model_selection import KFold


# Sirve para preparar el dataset para entrenar el modelo con el img. selection del labeler
def set_folds_db(db_path, table_name, k_folds, n_images):
    """
    Apply KFold logic to data in a SQLite table and save the results to a new table.

    Parameters:
    - db_path: Path to the SQLite database file.
    - table_name: Name of the source table to read data from.
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
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

    # Apply the logic from set_folds function
    if 'img_name' not in df.columns:
        raise ValueError("img_name column doesn't exist in the dataset.")

    df['k_fold'] = np.nan
    df['label_img'] = np.nan
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

# Example usage
# db_path = "/home/diego/Documents/yolov7-tracker/runs/detect/bytetrack_santos_dumont/santos_dumont_bbox.db"
# table_name = "bbox_raw"
# k_folds = 4
# n_images = 5
# set_folds_db(db_path, table_name, k_folds, n_images)

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