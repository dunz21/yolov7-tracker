import os
import cv2
import csv
import shutil
from shapely.geometry import LineString, Polygon, box
import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

def path_intersects_line(centroids, line):
    path = LineString(centroids)
    line = LineString(line)
    return path.intersects(line)

def point_side_of_line(point, line_start, line_end):
    line_vec = [line_end[0] - line_start[0], line_end[1] - line_start[1]]
    point_vec = [point[0] - line_start[0], point[1] - line_start[1]]
    cross_product = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    if cross_product > 0:
        return "In"
    elif cross_product < 0:
        return "Out"
    else:
        return "on the line"
### 1.- Obtener los posibles switchs IDs
def get_possible_switch_id(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM bbox_raw"
    df = pd.read_sql_query(query, conn)
    df['in_out_status'] = (df['distance_to_center'] >= 0).astype(int)
    df['in_out_status_label'] = df['in_out_status'].replace({1: 'Inside', 0: 'Outside'})
    conn.close()
    
    def detect_specific_transition(group): 
        group['status_change'] = group['in_out_status_label'].ne(group['in_out_status_label'].shift()).cumsum()
        segments_summary = group.groupby('status_change')['in_out_status_label'].agg(['first', 'count'])
        segments_list = list(segments_summary.itertuples(index=False, name=None))
        
        # Check if the first and last segment have the same 'in_out_status_label'
        if segments_list[0][0] != segments_list[-1][0]:
            return False  # Disregard if the beginning and end sequences don't match

        desired_patterns = [
            [('Inside', 10), ('Outside', 10), ('Inside', 10)],
            [('Outside', 10), ('Inside', 10), ('Outside', 10)]
        ]
        
        def pattern_exists(pattern):
            pattern_idx = 0
            for segment in segments_list:
                if pattern_idx >= len(pattern):
                    break
                if segment[0] == pattern[pattern_idx][0] and segment[1] >= pattern[pattern_idx][1]:
                    pattern_idx += 1
            return pattern_idx == len(pattern)
        
        return any(pattern_exists(pattern) for pattern in desired_patterns)
    
    behavior_presence_series = df.groupby('id').apply(detect_specific_transition)
    switch_risk_ids = behavior_presence_series[behavior_presence_series].index.tolist()
    return switch_risk_ids

### 2.- Obtener los IDs reales segun clusters, de todos los posibles
def get_switch_ids(ids, solider_df, silhouette_threshold=0.35, davies_bouldin_threshold=1.5, centroid_distance_threshold=0.8, plot=False):
    selected_ids_info = {}  # Changed to store more detailed info per ID
    
    features = solider_df.columns[solider_df.columns.str.startswith('Feature')]
    X = solider_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if plot:
        # Prepare for plotting
        n_ids = len(ids)
        cols = 2
        rows = n_ids // cols + (n_ids % cols > 0)
        plt.figure(figsize=(10, 4 * rows))
    
    for i, id_value in enumerate(ids, start=1):
        # Filter the rows for the current ID
        df_filtered = solider_df[solider_df['id'] == id_value]
        X_id = X_scaled[df_filtered.index]
        
        if len(X_id) < 2:  # Skip if not enough samples
            continue
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X_id)
        
        # Calculate the centroid distance
        centroids = kmeans.cluster_centers_
        centroid_distance = cosine_distances(centroids[0].reshape(1, -1), centroids[1].reshape(1, -1))[0][0]
        
        # Calculate scores
        silhouette = silhouette_score(X_id, labels)
        davies_bouldin = davies_bouldin_score(X_id, labels)
        
        if plot:
            # Reduce data to 2 dimensions for plotting
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_id)
            plt.subplot(rows, cols, i)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
            plt.title(f'ID: {id_value}\nSilhouette: {silhouette:.2f}, DB: {davies_bouldin:.2f}, Centroid Dist: {centroid_distance:.2f}')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
        
        # Append IDs and cluster info based on conditions
        if silhouette > silhouette_threshold and davies_bouldin < davies_bouldin_threshold or centroid_distance > centroid_distance_threshold:
            cluster_info = {0: [], 1: []}
            for label, img_name in zip(labels, df_filtered['img_name']):
                cluster_info[label].append(img_name)
            selected_ids_info[id_value] = cluster_info

    if plot:
        plt.tight_layout()
        plt.show()
    
    return selected_ids_info

### 3.- Obtener la separacion de las imagenes
def get_img_separation(switchs, db_path):
    ### Tengo un cluster de dos elementos. Pero desde ahi me quiero ir a la carpeta de la imagen y hacer una separacion en 2.
    ### Quiero hacer un corte
    list_ids = switchs.keys()
    conn = sqlite3.connect(db_path)
    query = "SELECT id, x1, y1, x2, y2, centroid_x, centroid_y, area, frame_number, overlap, distance_to_center, conf_score, img_name FROM bbox_raw where id in ({}) and img_name is not null order by id,frame_number".format(','.join([str(i) for i in list_ids]))
    df = pd.read_sql_query(query, conn)
    conn.close()
    # Initialize dictionary to store the results
    results_dict = {}
    for id in list_ids:
        id_df = df[df['id'] == id]
        
        # Sort the image names within each cluster
        switchs[id][0] = sorted(switchs[id][0], key=lambda x: int(x.split('_')[2]))
        switchs[id][1] = sorted(switchs[id][1], key=lambda x: int(x.split('_')[2]))
        
        frame_label_0 = int(switchs[id][0][0].split('_')[2])
        frame_label_1 = int(switchs[id][1][0].split('_')[2])
        
        # Split the data into two clusters based on frame_number
        if frame_label_0 < frame_label_1:
            cluster_A_df = id_df[id_df['frame_number'] < frame_label_1]
            cluster_B_df = id_df[id_df['frame_number'] >= frame_label_1]
        else:
            cluster_A_df = id_df[id_df['frame_number'] < frame_label_0]
            cluster_B_df = id_df[id_df['frame_number'] >= frame_label_0]
        
        # Function to determine directionality
        def determine_directionality(cluster_df):
            if cluster_df.empty:
                return None
            distances = cluster_df['distance_to_center'].values
            return "Out" if distances[-1] < distances[0] else "In"
        
        # Populate results for each cluster
        results_dict[id] = {
            "A": {
                "img_names": cluster_A_df['img_name'].tolist(),
                "direction": determine_directionality(cluster_A_df)
            },
            "B": {
                "img_names": cluster_B_df['img_name'].tolist(),
                "direction": determine_directionality(cluster_B_df)
            }
        }
    return results_dict
### 4.- Copiar las imagenes a las carpetas correspondientes
def process_and_copy_images(data, base_folder_path):
    for id, clusters in data.items():
        original_id_folder = os.path.join(base_folder_path, str(id))  # Convert id to str
        for cluster_label, info in clusters.items():
            direction = info['direction']
            new_id = int(id)
            while True:
                if direction == "In":
                    new_id -= 1
                else:  # direction == "Out"
                    new_id += 1
                
                new_id_folder = os.path.join(base_folder_path, str(new_id))
                if not os.path.exists(new_id_folder):
                    os.makedirs(new_id_folder)
                    break
            
            for img_name in info['img_names']:
                original_img_path = os.path.join(original_id_folder, img_name)
                new_img_name = img_name.replace(f'img_{id}_', f'img_{new_id}_').replace('Cross', direction)
                new_img_path = os.path.join(new_id_folder, new_img_name)
                
                # Copy and rename image
                shutil.copy(original_img_path, new_img_path)
    
def bbox_inside_any_polygon(polygons_points, bbox_tlbr):
    # Convert bbox from tlbr format to a Shapely box
    tl_x, tl_y, br_x, br_y = bbox_tlbr
    bbox = box(tl_x, tl_y, br_x, br_y)
    
    # Iterate over each polygon in the list
    for polygon_points in polygons_points:
        # Convert the current polygon points into a Shapely Polygon
        polygon = Polygon(polygon_points)
        
        # Check if the bbox is completely within the current polygon
        if polygon.contains(bbox):
            return True  # Return True if the bbox is inside any polygon
    
    return False  # Return False if the bbox is not inside any polygon
    
def guess_final_direction(arr, initial_value):
    """
    Removes all occurrences of initial_value from the beginning of the array until a different value is encountered.
    Then calculates the percentage of remaining elements in the array that are equal to initial_value.
    
    Parameters:
    - arr: List of strings, each being "In" or "Out"
    - initial_value: The initial value to remove and then calculate the percentage for ("In" or "Out")
    
    Returns:
    - Percentage of elements equal to initial_value in the remaining array
    """
    # Find the first index where the value is not initial_value
    first_different_index = None
    for i, value in enumerate(arr):
        if value != initial_value:
            first_different_index = i
            break
    
    # If all values are the same as initial_value, the remaining list is empty
    if first_different_index is None:
        return initial_value
    
    # Slice the array to remove the initial_values
    filtered_arr = arr[first_different_index:]
    
    # Count the occurrences of initial_value in the remaining array
    count_initial = filtered_arr.count(initial_value)
    
    # How many values of the initial value I have in the next part of the array
    # If its lower than 20% then I consider that the person is not going in the same direction
    if len(filtered_arr) > 0:
        percentage = (count_initial / len(filtered_arr)) * 100
        if percentage < 20:
            if initial_value == "In":
                return "Out"
            else:
                return "In"
    return initial_value
    



def save_image_based_on_sub_frame(num_frame, sub_frame, id, folder_name='images_subframe', direction=None, bbox=None):
    x1,y1,x2,y2,score = bbox
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    id_directory = os.path.join(f"{folder_name}", str(id))
    if not os.path.exists(id_directory):
        os.makedirs(id_directory)
    image_name = f"img_{id}_{num_frame}_{direction}_{x1}_{y1}_{x2}_{y2}_{score:.2f}.png"
    save_path = os.path.join(id_directory, image_name)
    cv2.imwrite(save_path, sub_frame)
    return image_name

def save_csv_bbox_alternative(personImage, filepath='',folder_name='', direction=''):
    file_exists = os.path.isfile(filepath)

    # Open the file in append mode ('a') if it exists, otherwise in write mode ('w')
    with open(filepath, 'a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file is being created for the first time
        if not file_exists:
            writer.writerow(['id', 'x1', 'y1', 'x2', 'y2', 'centroid_x', 'centroid_y', 'area', 'frame_number', 'overlap', 'distance_to_center', 'conf_score','img_name'])

        for index, img in enumerate(sorted(personImage.list_images, key=lambda x: x.frame_number)):
            image_name = ''
            if img.img_frame is not None:
                image_name = save_image_based_on_sub_frame(img.frame_number, img.img_frame, personImage.id, folder_name=folder_name, direction=direction, bbox=img.bbox)                
            x1, y1, x2, y2, conf_score = img.bbox
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            overlap_rounded = round(img.overlap, 2)
            distance_to_center_rounded = round(img.distance_to_center, 2)
            conf_score_rounded = round(conf_score, 2)
            writer.writerow([personImage.id, int(x1), int(y1), int(x2), int(y2), int(centroid_x), int(centroid_y), area, img.frame_number, overlap_rounded, distance_to_center_rounded, conf_score_rounded, image_name])

            