# For plotting and data transformation
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import zip_longest


import pandas as pd
import numpy as np
# For file and directory operations
import os
import glob
import csv

# For image processing
from PIL import Image
from sklearn.cluster import KMeans

# For clustering and evaluation
from sklearn.metrics import silhouette_score, davies_bouldin_score
import base64
from utils.solider import in_out_status,seconds_to_time,solider_result,custom_threshold_analysis
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine as cosine_distance

# NO ES IMPORTANTE
def evaluate_clustering(features, image_names, num_clusters=2):
    """
    Evaluates the clustering performance using KMeans algorithm and returns the silhouette and 
    Davies-Bouldin scores, along with the clustered images.

    :param features: A 2D numpy array where each row represents the feature vector of an image.
    :param image_names: A list of image names corresponding to each feature vector.
    :param num_clusters: The number of clusters to form. Default is 2.
    :return: A tuple containing the silhouette score, Davies-Bouldin score, and a list of tuples with
             image names and their corresponding cluster labels.
    """
    # Create KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Fit the model
    kmeans.fit(features)

    # Predict the cluster for each image
    cluster_labels = kmeans.labels_

    # Calculate silhouette score
    silhouette_avg = silhouette_score(features, cluster_labels)

    # Calculate Davies-Bouldin score
    davies_bouldin_avg = davies_bouldin_score(features, cluster_labels)

    # Pairing image names with their respective cluster labels
    # clustered_images = list(zip(image_names, cluster_labels))

    return silhouette_avg, davies_bouldin_avg#, clustered_images

def _chunk_array(array, chunk_size):
    chunks = []
    for i in range(0, len(array), chunk_size):
        chunks.append(array[i:i + chunk_size])
    return chunks

def _image_formatter(img_file):
    try:
        with open(img_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
            return f'<img width="125" src="data:image/png;base64,{encoded_string}">'
    except FileNotFoundError:
        return "Image not found"

def _save_solider_csv_by_chunks(features_array, images_names, filename):
    # Ensure the filename ends with '.csv'
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)

    # Open the file in append mode
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)

        # Write headers if the file is new
        if not file_exists:
            writer.writerow(['Name', 'ID', 'Direction'] + [f'Feature_{i+1}' for i in range(len(features_array[0]))])

        # Write data
        for image_name, features in zip(images_names, features_array):
            id = image_name.split('_')[1]
            direction = image_name.split('_')[3]
            feature_list = [str(f) for f in features]  # Convert features to a list of strings
            row_data = [image_name, id, direction] + feature_list
            writer.writerow(row_data)

def _parseDataSolider(dataframe_solider):
    for col in dataframe_solider.columns[3:]:  
        dataframe_solider[col] = dataframe_solider[col].astype(float)

    dataframe_solider['ID'] = dataframe_solider['ID'].astype(int)

    for id in dataframe_solider['ID'].unique():
        directions = dataframe_solider.loc[dataframe_solider['ID'] == id, 'Direction'].tolist()
        new_direction = in_out_status(directions)
        dataframe_solider.loc[dataframe_solider['ID'] == id, 'Direction'] = new_direction
    return dataframe_solider

# 0.- Get Folders
def get_folders(parent_folder, limit=None):
    # List all entries in the parent folder
    parent_folder = os.path.abspath(parent_folder)
    entries = os.listdir(parent_folder)

    
    # Filter out the subfolders and get their full paths
    subfolder_paths = [os.path.join(parent_folder, entry) for entry in entries if os.path.isdir(os.path.join(parent_folder, entry))]
    
    # Sort the subfolders by their names converted to integers
    subfolder_paths.sort(key=lambda x: int(os.path.basename(x)))

    if limit is not None:
        return subfolder_paths[:limit]
    return subfolder_paths

# 1.- Analisis pureza 
def folder_analysis(folders):
    result = {
    'In': [],
    'Out': [],
    'InOut': []
    }
    count = 0

    for folder in folders:
        list_images = os.listdir(folder)
        if len(list_images) == 0:
            print(f"Folder {folder} is empty")
        images_status = [img.split('_')[3] for img in list_images]
        image_statuses = in_out_status(images_status,1)
        if folder.split('/')[-1] == '149':
            print('asdf')
        if 'In' == image_statuses:
            result['In'].append(folder.split('/')[-1])
        elif 'Out' == image_statuses:
            result['Out'].append(folder.split('/')[-1])
        else:
            result['InOut'].append(folder.split('/')[-1])
        
        count += 1

    total_folders = count
    total_in = len(result['In'])
    total_out = len(result['Out'])
    total_in_out = len(result['InOut'])

    return total_folders,total_in,total_out,total_in_out,result

# 2.- Generate Feature Solider and save to csv
def save_folders_to_solider_csv(list_folders_in_out, name_csv):
    full_path = []
    for folder in list_folders_in_out:
        entries = os.listdir(folder)
        if len(entries) == 0:
            print(f"Folder {folder} is empty")
            continue
        full_path.append(folder)
    chunks = _chunk_array(full_path, 40)

    all_data = []

    for chunk in chunks:
        features_array, image_names = solider_result(chunk)
        
        # Process each image and its features
        for image_name, features in zip(image_names, features_array):
            id = image_name.split('_')[1]
            direction = image_name.split('_')[3]
            feature_list = [str(f) for f in features]  # Convert features to a list of strings
            row_data = [image_name, id, direction] + feature_list
            all_data.append(row_data)

        # Optionally, you can also write to CSV in chunks
        _save_solider_csv_by_chunks(features_array, image_names, name_csv)

    # Define the DataFrame columns
    columns = ['Name', 'ID', 'Direction'] + [f'Feature_{i+1}' for i in range(len(features_array[0]))]

    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data, columns=columns)
    df = _parseDataSolider(df)
    return df

# 2.- Get Feature Solider
def get_feature_img_csv(start_row=0, end_row=900, csv_file='solider_result.csv'):
    chunksize = end_row - start_row
    df = pd.read_csv(csv_file, skiprows=range(1, start_row + 1), nrows=chunksize)
    df = _parseDataSolider(df)
    return df

# 3.- Distances with silhoutte score
def generate_in_out_distance_plot_csv(features, plot=False, csv_file_path=None, distance='kmeans'):
    # Separate 'In' and 'Out' data
    df_in = features[features['Direction'] == 'In']
    df_out = features[features['Direction'] == 'Out']

    # Get unique IDs for 'In' and 'Out'
    unique_ids_in = df_in['ID'].unique()
    unique_ids_out = df_out['ID'].unique()

    # Initialize silhouette_avg DataFrame
    silhouette_avg = pd.DataFrame(index=unique_ids_in, columns=unique_ids_out)
    silhouette_avg[:] = np.nan  # Fill with NaN

    mask = np.zeros_like(silhouette_avg, dtype=bool)

    for j, id_out in enumerate(unique_ids_out):
        for i, id_in in enumerate(unique_ids_in):
            if id_in >= id_out:  # For example, compare only if row ID is less than column ID
                mask[i, j] = True
                continue
            data_in = df_in[df_in['ID'] == id_in].iloc[:, 3:]
            data_out = df_out[df_out['ID'] == id_out].iloc[:, 3:]
            if not data_in.empty and not data_out.empty:

                mean_features_in = np.mean(data_in, axis=0)
                mean_features_out = np.mean(data_out, axis=0)
                if distance == 'euclidean':
                    euclidean_distance = euclidean(mean_features_in, mean_features_out)
                    silhouette_avg.at[id_in, id_out] = round(euclidean_distance,2)
                elif distance == 'cosine':
                    cosine_dist = cosine_distance(mean_features_in, mean_features_out)
                    silhouette_avg.at[id_in, id_out] = round(cosine_dist,2)
                elif distance == 'kmeans':
                    #KMEANS
                    data_cluster = pd.concat([data_in, data_out])
                    kmeans = KMeans(n_clusters=2, random_state=42).fit(data_cluster)
                    if len(data_cluster) > 1:
                        unique_labels = np.unique(kmeans.labels_)
                        if 1 < len(unique_labels) < len(data_cluster):
                            score = silhouette_score(data_cluster, kmeans.labels_)
                            silhouette_avg.at[id_in, id_out] = round(score,2)

    if csv_file_path is not None:
        silhouette_avg.to_csv(csv_file_path)

    if plot:
        silhouette_avg_numeric = silhouette_avg.applymap(pd.to_numeric, errors='coerce')
        plt.figure(figsize=(20, 16))
        sns.heatmap(silhouette_avg_numeric, annot=True, cmap='coolwarm', mask=mask)
        plt.title('Silhouette Scores of K-Means Clustering')
        plt.xlabel('Out IDs')
        plt.ylabel('In IDs')
        plt.show()

    return silhouette_avg

# 4.- Get Match
def get_match_pair(silhouette_avg, row_or_col='row'):
    """
        Esta funcion recorre el heatmap para buscar segun una salida (columna) la entrada que mas
        se le parezca. Y eso lo hace buscando por columna el valor mas bajo pero ese valor bajo
        tiene que estar alejado del segun valor mas bajo y tercero por eso se ocupa el Z Score

        Tiene la opcion de entregarte el menor valor de todas las columnas o filas. Y te entrega
        los pares encontrados
    """
    DELETE_PREV_ROWS = True
    DELETE_PREV_COL = True
    THRESHOLD = 93
    pair_row_col = {
        'row': [],
        'col': [],
        'value': []
    }
    silhouette_avg2 = silhouette_avg.copy()
    if row_or_col == 'row':
        silhouette_avg2 = silhouette_avg2.T

    for col in silhouette_avg2.columns:
        col_data = np.array(silhouette_avg2[col].dropna())

        if col_data.size == 0:
            continue

        # CASO BORDE
        if len(col_data) == 1:
            if np.min(col_data) < 0.33:
                thereshold = {
                    "lowest_value": np.min(col_data),
                    "second_lowest": 0,
                    "percentage_difference": 0,
                    "is_considerably_lower": True
                }
            else: 
                thereshold = {
                    "lowest_value": np.min(col_data),
                    "second_lowest": 0,
                    "percentage_difference": 0,
                    "is_considerably_lower": False
                }
        else:
            thereshold = custom_threshold_analysis(col_data, threshold=THRESHOLD)

        if thereshold['is_considerably_lower'] and thereshold['lowest_value'] < 0.5:
            # Estoy casi seguro que es el mejor
            min_value_col = np.min(col_data)
            index_row = silhouette_avg2[silhouette_avg2[col] == min_value_col].index[0]
            pair_row_col['row'].append(index_row)
            pair_row_col['col'].append(col)
            pair_row_col['value'].append(min_value_col)
            if DELETE_PREV_ROWS:
                silhouette_avg2 = silhouette_avg2.drop(index_row)
            if DELETE_PREV_COL:
                silhouette_avg2 = silhouette_avg2.drop(col, axis=1)
    return pair_row_col,silhouette_avg2

# 5.- Export to HTML
def export_to_html(list_image_in, list_in, list_image_out, list_out, total_folders, total_in, total_out, total_in_out, scoreOut,  filename='export.html',frame_rate=15):
    # Check if lists are of the same length
    if not (len(list_image_in) == len(list_in) == len(list_image_out) == len(list_out)):
        raise ValueError("All lists must be of the same length.")

    # Create DataFrame
    df = pd.DataFrame({
        'ImageIn': list_image_in,
        'ImageOut': list_image_out,
        'FolderIn': list_in,
        'FolderOut': list_out
    })

    # Calculate TimeDiff
    df['TimeDiff'] = df.apply(lambda row: seconds_to_time((int(row['ImageOut'].split('_')[2]) - int(row['ImageIn'].split('_')[2]))/ frame_rate), axis=1)

    # Format images
    df['ImageIn'] = df['ImageIn'].apply(_image_formatter)
    df['ImageOut'] = df['ImageOut'].apply(_image_formatter)


    # Create summary DataFrame
    summary_data = {
        'Metric': ['Total folders', 'Total In', 'Total Out', 'Total In_Out','Score'],
        'Value': [total_folders,f"{total_in} {(total_in / total_folders) * 100:.2f}%", f"{total_out} {(total_out / total_folders) * 100:.2f}%", f"{total_in_out} {(total_in_out / total_folders) * 100:.2f}%",scoreOut]
    }
    summary_df = pd.DataFrame(summary_data)

    # Convert each DataFrame to HTML
    html_df1 = summary_df.to_html(escape=False, index=False)
    html_df2 = df.to_html(escape=False, index=False)

    # Concatenate HTML strings with a separator
    combined_html = html_df1 + "<br><hr><br>" + html_df2
    # Convert DataFrame to HTML and write to file
    with open(filename, 'w') as file:
        file.write(combined_html)

# FINAL PIPLELINE
def getFinalScore(folder_name,solider_file='solider_results.csv',silhoutte_file='silhoutte_avg.csv', html_file='export.html',distance_method='kmeans'):
    list_folders = get_folders(folder_name)

    base_path = os.path.dirname(list_folders[0])

    total_folders,total_in,total_out,total_in_out,result = folder_analysis(list_folders)

    if os.path.exists(solider_file):
        solider_df = get_feature_img_csv(csv_file=solider_file,start_row=0,end_row=4000)
    else:
        list_folders_in_out = [os.path.join(base_path, folder) for folder in result['In']] + [os.path.join(base_path, folder) for folder in result['Out']]
        solider_df = save_folders_to_solider_csv(list_folders_in_out,solider_file)

    if os.path.exists(silhoutte_file):
        silhoutte_df = pd.read_csv(silhoutte_file, index_col=0)
    else:
        silhoutte_df = generate_in_out_distance_plot_csv(solider_df, plot=False, csv_file_path=silhoutte_file, distance=distance_method)

    pair_result, silhoutte_results = get_match_pair(silhouette_avg=silhoutte_df, row_or_col='col')

    if len(pair_result['col']) == 0 or len(pair_result['row']) == 0:
        print('No Match Found')
        return

    scoreOut = f"{len(pair_result['col'])}/{len(result['Out'])} ({len(pair_result['col']) / len(result['Out'])*100:.2f}%)"
    

    list_image_in = [os.path.join(base_path,str(row),os.listdir(os.path.join(base_path,str(row)))[0]) for row in pair_result['row']]
    list_image_out = [os.path.join(base_path,str(col),os.listdir(os.path.join(base_path,str(col)))[0]) for col in pair_result['col']]

    export_to_html(list_image_in, pair_result['row'], list_image_out, pair_result['col'], total_folders, total_in, total_out, total_in_out,scoreOut, html_file)




def _count_images(folder):
    return len(os.listdir(folder))

def _calculate_silhouette_score(features,cluster):
    kmeans = KMeans(n_clusters=cluster, random_state=42)
    kmeans.fit(features)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(features, cluster_labels)
    return f"{silhouette_avg:.2f}"

# Extra for debugging, 
def export_images_in_out_to_html(csv_file_silhoutte, csv_solider, base_path, filename='export.html'):
    cluster = 2
    silhoutte_df = pd.read_csv(csv_file_silhoutte, index_col=0)
    solider_df = pd.read_csv(csv_solider)
    list_in = silhoutte_df.index.values.tolist()
    list_out = silhoutte_df.columns.values.tolist()

    rows = []
    for in_folder, out_folder in zip_longest(list_in, list_out):
        # Handle the case where in_folder or out_folder is None
        in_folder_path = os.path.join(base_path, str(in_folder)) if in_folder is not None else None
        out_folder_path = os.path.join(base_path, str(out_folder)) if out_folder is not None else None

        # Load features from the dataframe
        features_in = solider_df[solider_df['ID'] == int(in_folder)].iloc[:, 3:] if in_folder is not None else None
        features_out = solider_df[solider_df['ID'] == int(out_folder)].iloc[:, 3:] if out_folder is not None else None

        # Get the first image, count images, and calculate silhouette score
        first_image_in = os.path.join(in_folder_path, os.listdir(in_folder_path)[0]) if in_folder_path and os.path.isdir(in_folder_path) else "No Image"
        first_image_out = os.path.join(out_folder_path, os.listdir(out_folder_path)[0]) if out_folder_path and os.path.isdir(out_folder_path) else "No Image"

        in_image_count = _count_images(in_folder_path) if in_folder_path and os.path.isdir(in_folder_path) else 0
        out_image_count = _count_images(out_folder_path) if out_folder_path and os.path.isdir(out_folder_path) else 0

        in_silhouette = _calculate_silhouette_score(features_in, cluster) if in_folder_path and in_image_count > cluster else 'N/A'
        out_silhouette = _calculate_silhouette_score(features_out, cluster) if out_folder_path and out_image_count > cluster else 'N/A'

        row = {
            'ImageIn': _image_formatter(first_image_in) if in_folder_path else "No Image",
            'ID In': in_folder if in_folder is not None else "N/A",
            '# & Silhouette Score In': f"{in_image_count}({in_silhouette})",
            'ImageOut': _image_formatter(first_image_out) if out_folder_path else "No Image",
            'ID Out': out_folder if out_folder is not None else "N/A",
            '# & Silhouette Score Out': f"{out_image_count}({out_silhouette})"
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert DataFrame to HTML and write to file
    html_content = df.to_html(escape=False, index=False)
    with open(filename, 'w') as file:
        file.write(html_content)