import os
import sqlite3
import pandas as pd
import shutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from utils.pipeline import get_folders, save_folders_to_solider_csv
from utils.tools import convert_csv_to_sqlite


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
def process_and_copy_images(data, base_folder_path, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        for id, clusters in data.items():
            original_id_folder = os.path.join(base_folder_path, str(id))
            for cluster_label, info in clusters.items():
                direction = info['direction']
                new_id = int(id)
                while True:
                    new_id = new_id - 1 if direction == "In" else new_id + 1
                    new_id_folder = os.path.join(base_folder_path, str(new_id))
                    if not os.path.exists(new_id_folder):
                        os.makedirs(new_id_folder)
                        break
                try:
                    # Extract last frame number more robustly
                    last_frame = int(info['img_names'][-1].split('_')[2].split('.')[0])
                    
                    # Database transaction for updating IDs
                    cursor.execute("BEGIN")
                    cursor.execute("UPDATE bbox_raw SET id = ? WHERE id = ? AND frame_number <= ?", (new_id, id, last_frame))
                    cursor.execute("COMMIT")
                    
                    # Update image names and copy files
                    for img_name in info['img_names']:
                        original_img_path = os.path.join(original_id_folder, img_name)
                        new_img_name = img_name.replace(f'img_{id}_', f'img_{new_id}_').replace('Cross', direction)
                        cursor.execute("BEGIN")
                        cursor.execute("UPDATE bbox_raw SET img_name = ? WHERE img_name = ?", (new_img_name, img_name))
                        cursor.execute("COMMIT")
                        new_img_path = os.path.join(new_id_folder, new_img_name)
                        shutil.copy(original_img_path, new_img_path)
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    print(f"Error during processing: {e}")
                    
            # Ensure deletion is within a transaction
            cursor.execute("BEGIN")
            cursor.execute("DELETE FROM bbox_raw WHERE id = ?", (id,))
            cursor.execute("COMMIT")
            
            shutil.rmtree(original_id_folder)
    except Exception as e:
        print(f"Unhandled error: {e}")
    finally:
        conn.close()  # Ensure the connection is closed even if an error occurs
        
        
        
def switch_id_corrector_pipeline(db_path='', base_folder_path='',weights='',model_name='solider', plot=False):
    silhouette_threshold=0.35
    davies_bouldin_threshold=1.5
    centroid_distance_threshold=0.8
    # Step 1: Get the possible switch IDs
    possible_switch_ids = get_possible_switch_id(db_path)
    
    list_folders = get_folders(base_folder_path)
    list_folders = [path for path in list_folders if int(path.split('/')[-1]) in possible_switch_ids]
    solider_df = save_folders_to_solider_csv(list_folders_in_out= list_folders,weights=weights ,model_name=model_name)
    
    # Step 2: Get the switch IDs based on clustering
    switch_ids = get_switch_ids(possible_switch_ids, solider_df, silhouette_threshold, davies_bouldin_threshold, centroid_distance_threshold, plot)
    
    # Step 3: Get the image separation for each switch ID
    img_separation = get_img_separation(switch_ids, db_path)
    
    # Step 4: Process and copy the images to the corresponding folders
    process_and_copy_images(img_separation, base_folder_path, db_path)
    
    return img_separation



if __name__ == '__main__':
    folder_path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_03_conce_debug_switch_id/imgs_conce_debug'
    db_path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_03_conce_debug_switch_id/conce_debug_bbox.db'
    csv_path = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_03_conce_debug_switch_id/conce_debug_bbox.csv'
    convert_csv_to_sqlite(csv_file_path=csv_path, db_file_path=db_path, table_name='bbox_raw')

    ### 1.- Get the possible switch IDs
    ids = get_possible_switch_id(db_path)

    ##### SOLIDER #####
    list_folders = get_folders(folder_path)
    folders_to_include = ids
    list_folders = [path for path in list_folders if int(path.split('/')[-1]) in folders_to_include]
    solider_df = save_folders_to_solider_csv(list_folders_in_out= list_folders,weights='model_weights.pth',model_name='solider')
    ##### SOLIDER #####

    ### 2.- Get the switch IDs
    switchs_dict = get_switch_ids(ids, solider_df, plot=False)
    #cluster_info = {0: [], 1: []}

    ### 3.- Get the separation of the images
    clusters_dict = get_img_separation(switchs_dict, db_path)
    # results_dict[id] = {
    # 			"A": {
    # 				"img_names": cluster_A_df['img_name'].tolist(),
    # 				"direction": determine_directionality(cluster_A_df)
    # 			},
    # 			"B": {
    # 				"img_names": cluster_B_df['img_name'].tolist(),
    # 				"direction": determine_directionality(cluster_B_df)
    # 			}
    # 		}
    ### 4.- Process and copy the images
    process_and_copy_images(clusters_dict, folder_path,db_path)

    #dict_keys([330, 682, 1848])