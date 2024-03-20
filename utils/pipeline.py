# For plotting and data transformation
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import zip_longest
import torch

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
from utils.solider import in_out_status,seconds_to_time,model_selection,custom_threshold_analysis
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine as cosine_distance
import time
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import sqlite3

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
            writer.writerow(['img_name', 'id', 'direction'] + [f'Feature_{i+1}' for i in range(len(features_array[0]))])

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

    dataframe_solider['id'] = dataframe_solider['id'].astype(int)

    for id in dataframe_solider['id'].unique():
        directions = dataframe_solider.loc[dataframe_solider['id'] == id, 'direction'].tolist()
        new_direction = in_out_status(directions)
        dataframe_solider.loc[dataframe_solider['id'] == id, 'direction'] = new_direction
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
def save_folders_to_solider_csv(list_folders_in_out, name_csv, weights='', model='', save_to_db=None):
    full_path = []
    for folder in list_folders_in_out:
        entries = os.listdir(folder)
        if len(entries) == 0:
            print(f"Folder {folder} is empty")
            continue
        full_path.append(folder)
    chunks = _chunk_array(full_path, 20)

    all_data = []

    for chunk in chunks:
        features_array, image_names = model_selection(name=model, folder_path=chunk, weights=weights)
        
        # Process each image and its features
        for image_name, features in zip(image_names, features_array):
            id = image_name.split('_')[1]
            direction = image_name.split('_')[3].replace('.jpg','') # Assuming image names end with '.jpg'
            feature_list = [str(f) for f in features]  # Convert features to a list of strings
            row_data = [image_name, id, direction] + feature_list
            all_data.append(row_data)

        # Optionally, you can also write to CSV in chunks
        _save_solider_csv_by_chunks(features_array, image_names, name_csv)

    # Define the DataFrame columns
    columns = ['img_name', 'id', 'direction'] + [f'feature_{i+1}' for i in range(len(features_array[0]))]

    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data, columns=columns)
    df = _parseDataSolider(df)

    # If save_to_db is provided, save DataFrame to SQLite database
    if save_to_db:
        conn = sqlite3.connect(name_csv.replace('.csv','.db'))
        df.to_sql('features', conn, if_exists='replace', index=False)
        conn.close()

    return df

# 2.- Get Feature Solider
def get_feature_img_csv(start_row=0, end_row=900, csv_file='features.csv'):
    # chunksize = end_row - start_row
    # df = pd.read_csv(csv_file, skiprows=range(1, start_row + 1), nrows=chunksize)
    # df = _parseDataSolider(df)
    
    df = pd.read_csv(csv_file)
    df = _parseDataSolider(df)
    return df

# 3.- Distances with silhoutte score
def generate_in_out_distance_plot_csv(features, plot=False, csv_file_path=None, distance='euclidean'):
    # Separate 'In' and 'Out' data
    df_in = features[features['direction'] == 'In']
    df_in = df_in.drop_duplicates(subset=['id'])
    df_in = df_in.iloc[:, 3:]

    df_out = features[features['direction'] == 'Out']
    df_out = df_out.drop_duplicates(subset=['id'])
    df_out = df_out.iloc[:, 3:]

    # Convert to numpy arrays for efficient computation
    X_in = df_in.to_numpy()
    X_out = df_out.to_numpy()

    # Compute distances based on the specified metric
    if distance == 'euclidean':
        df_in = torch.tensor(df_in.to_numpy())
        df_out = torch.tensor(df_out.to_numpy())
        X_in = 1. * df_in / (torch.norm(df_in, 2, dim = -1, keepdim=True).expand_as(df_in) + 1e-12)
        X_out = 1. * df_out / (torch.norm(df_out, 2, dim = -1, keepdim=True).expand_as(df_out) + 1e-12)
        X_in = X_in.numpy()
        X_out = X_out.numpy()
        dist_matrix = cdist(X_in, X_out, metric='euclidean')
    elif distance == 'cosine':
        dist_matrix = cdist(X_in, X_out, metric='cosine')
    else:
        raise ValueError("Unsupported distance metric. Use 'euclidean' or 'cosine'.")

    # Convert distance matrix to DataFrame for easier handling
    unique_ids_in = features[features['direction'] == 'In']['id'].unique()
    unique_ids_out = features[features['direction'] == 'Out']['id'].unique()
    # distance_matrix = pd.DataFrame(dist_matrix, index=unique_ids_in, columns=unique_ids_out)

    # Create a matrix of zeros with the same shape as `dist_matrix`
    adjusted_matrix = np.zeros_like(dist_matrix)
    # Fill in the adjusted matrix with dist_matrix values where row index <= column index
    for i, row_id in enumerate(unique_ids_in):
        for j, col_id in enumerate(unique_ids_out):
            if row_id <= col_id:
                # If the row index is NOT greater than the column index, copy the value
                adjusted_matrix[i, j] = dist_matrix[i, j]
    # Convert the adjusted matrix back to a DataFrame
    distance_matrix = pd.DataFrame(adjusted_matrix, index=unique_ids_in, columns=unique_ids_out)
    distance_matrix.replace(0, np.nan, inplace=True)

    # Optionally save to CSV
    if csv_file_path is not None:
        distance_matrix.to_csv(csv_file_path)

    # Optionally plot
    if plot:
        plt.figure(figsize=(20, 16))
        sns.heatmap(distance_matrix, annot=True, cmap='coolwarm')
        plt.title(f'{distance.capitalize()} Distance between In and Out IDs')
        plt.xlabel('Out IDs')
        plt.ylabel('In IDs')
        plt.show()

    return distance_matrix

# 4.- Get Match
def get_match_pair(distance_data, row_or_col='row'):
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
    distance_data_output = distance_data.copy()
    if row_or_col == 'row':
        distance_data_output = distance_data_output.T

    for col in distance_data_output.columns:
        col_data = np.array(distance_data_output[col].dropna())

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
            index_row = distance_data_output[distance_data_output[col] == min_value_col].index[0]
            pair_row_col['row'].append(index_row)
            pair_row_col['col'].append(col)
            pair_row_col['value'].append(min_value_col)
            if DELETE_PREV_ROWS:
                distance_data_output = distance_data_output.drop(index_row)
            if DELETE_PREV_COL:
                distance_data_output = distance_data_output.drop(col, axis=1)
    return pair_row_col,distance_data_output

# 4.5- Get Match
def create_sorted_matrix(df, k, base_path):
    # A list to store the results
    result_list = []
    
    # Iterate over each column (person out)
    for col in df.columns:  # Skipping the first column as it's the row identifier
        # Sort the column to get k smallest distances and their corresponding row indices (ignoring zeros)
        row = []
        sorted_indices = df[df[col] > 0][col].nsmallest(k).index.tolist()
        sorted_distances = df.loc[sorted_indices, col].tolist()
        
        # Add the image path and id for the column itself
        image_out = os.path.join(base_path,str(col),os.listdir(os.path.join(base_path,str(col)))[0])
        row.append({'img': f'{image_out}', 'id': col})
        
        # Add the sorted distances and their corresponding row indices
        for i in range(min(k,len(sorted_distances))):
            image_in = os.path.join(base_path,str(sorted_indices[i]),os.listdir(os.path.join(base_path,str(sorted_indices[i])))[0])
            time_diff = int((int(image_out.split('/')[-1].split('_')[2]) - int(image_in.split('/')[-1].split('_')[2])) / 15)
            if time_diff < 0:
                time_diff = seconds_to_time(0)
            else:
                time_diff = seconds_to_time(time_diff)
            obj_img = {'img': f'{image_in}', 'id': sorted_indices[i], 'distance': f"{sorted_distances[i]:.2f}", 'time_diff': time_diff}
            row.append(obj_img)
        result_list.append(row)
    
    return result_list

def export_ranking_html(result_list, filename='export.html'):
    # Create a list of HTML strings for each row
    html_rows = []
    for row in result_list:
        # Create a list of HTML strings for each cell in the row
        html_cells = []
        for cell in row:
            # Create the HTML string for the cell
            cell_html = f'<td><img height="200" width="125" src="{cell["img"]}"><br>ID: {cell["id"]}<br>Distance: {cell.get("distance", "N/A")} <br>Time: {cell.get("time_diff", "N/A")} </td>'
            html_cells.append(cell_html)
        
        # Join the HTML strings for the cells
        html_row = '<tr>' + ''.join(html_cells) + '</tr>'
        html_rows.append(html_row)
    
    # Join the HTML strings for the rows
    html_content = '<table>' + ''.join(html_rows) + '</table>'
    
    # Write the HTML content to a file
    with open(filename, 'w') as file:
        file.write(html_content)

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
    # df['TimeDiff'] = df.apply(lambda row: seconds_to_time((int(row['ImageOut'].split('/')[-1].split('_')[2]) - int(row['ImageIn'].split('/')[-1].split('_')[2]))/ frame_rate), axis=1)


    def debug_time_diff(row):
        try:
            # Extracting and calculating the time difference
            time_diff = int((int(row['ImageOut'].split('/')[-1].split('_')[2]) - int(row['ImageIn'].split('/')[-1].split('_')[2])) / frame_rate)
            # Converting to time format
            if(time_diff < 0):
                return seconds_to_time(0)
            return seconds_to_time(time_diff)
        except OverflowError as e:
            # Printing the problematic values
            print(f"Error in row: {row}")
            print(f"Start: {int(row['ImageOut'].split('/')[-1].split('_')[2])} end : {int(row['ImageIn'].split('/')[-1].split('_')[2])} {time_diff}")
            print(f"ImageOut: {row['ImageOut']}, ImageIn: {row['ImageIn']}, frame_rate: {frame_rate}")
            raise e

    # Apply the debug function to each row
    df['TimeDiff'] = df.apply(debug_time_diff, axis=1)

    df['Start'] = df.apply(lambda row: seconds_to_time((int(row['ImageIn'].split('/')[-1].split('_')[2])// frame_rate)), axis=1)
    df['End'] = df.apply(lambda row: seconds_to_time((int(row['ImageOut'].split('/')[-1].split('_')[2])// frame_rate)), axis=1)

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


def get_features_from_model(model, folder_path='',features_file='features.csv', weights=''):
    list_folders = get_folders(folder_path)
    _,_,_,_,result = folder_analysis(list_folders)
    base_path = os.path.dirname(list_folders[0])
    list_folders_in_out = [os.path.join(base_path, folder) for folder in result['In']] + [os.path.join(base_path, folder) for folder in result['Out']]
    features = save_folders_to_solider_csv(list_folders_in_out,features_file,weights,model=model,save_to_db=True)
    return features

# FINAL PIPLELINE
def getFinalScore(folder_name,weights='',model='',features_file='features.csv',distance_file='', html_file='export.html',distance_method='euclidean',frame_rate=15):
    list_folders = get_folders(folder_name)

    base_path = os.path.dirname(list_folders[0])

    total_folders,total_in,total_out,total_in_out,result = folder_analysis(list_folders)
    if os.path.exists(features_file):
        solider_df = get_feature_img_csv(csv_file=features_file,start_row=0,end_row=4000)
    else:
        list_folders_in_out = [os.path.join(base_path, folder) for folder in result['In']] + [os.path.join(base_path, folder) for folder in result['Out']]
        solider_df = save_folders_to_solider_csv(list_folders_in_out,features_file,weights,model=model,save_to_db=True)
        

    distance_data = generate_in_out_distance_plot_csv(solider_df, plot=False, csv_file_path=distance_file, distance=distance_method)

    test = create_sorted_matrix(distance_data, 5, base_path)

    pair_result, distance_data_output = get_match_pair(distance_data=distance_data, row_or_col='col')

    if len(pair_result['col']) == 0 or len(pair_result['row']) == 0:
        print('No Match Found')
        return

    scoreOut = f"{len(pair_result['col'])}/{len(result['Out'])} ({len(pair_result['col']) / len(result['Out'])*100:.2f}%)"
    

    list_image_in = [os.path.join(base_path,str(row),os.listdir(os.path.join(base_path,str(row)))[0]) for row in pair_result['row']]
    list_image_out = [os.path.join(base_path,str(col),os.listdir(os.path.join(base_path,str(col)))[0]) for col in pair_result['col']]
    # export_ranking_html(test, html_file)
    export_to_html(list_image_in=list_image_in, list_in=pair_result['row'],list_image_out=list_image_out, list_out=pair_result['col'], total_folders=total_folders, total_in=total_in,total_out=total_out,total_in_out=total_in_out, scoreOut=scoreOut,filename=html_file,frame_rate=frame_rate)


##### OTHER THINGS ####

# Extra for debugging. Show all images indepent if there is a match
def export_images_in_out_to_html(csv_file_distance, csv_solider, base_path, filename='export.html',frame_rate=15):
    cluster = 2
    silhoutte_df = pd.read_csv(csv_file_distance, index_col=0)
    solider_df = pd.read_csv(csv_solider)
    list_in = silhoutte_df.index.values.tolist()
    list_out = silhoutte_df.columns.values.tolist()

    rows = []
    for in_folder, out_folder in zip_longest(list_in, list_out):
        # Handle the case where in_folder or out_folder is None
        in_folder_path = os.path.join(base_path, str(in_folder)) if in_folder is not None else None
        out_folder_path = os.path.join(base_path, str(out_folder)) if out_folder is not None else None

        # Load features from the dataframe
        features_in = solider_df[solider_df['id'] == int(in_folder)].iloc[:, 3:] if in_folder is not None else None
        features_out = solider_df[solider_df['id'] == int(out_folder)].iloc[:, 3:] if out_folder is not None else None

        # Get the first image, count images, and calculate silhouette score
        first_image_in = os.path.join(in_folder_path, os.listdir(in_folder_path)[0]) if in_folder_path and os.path.isdir(in_folder_path) else "No Image"
        first_image_out = os.path.join(out_folder_path, os.listdir(out_folder_path)[0]) if out_folder_path and os.path.isdir(out_folder_path) else "No Image"

        start_time = 0 if first_image_in == "No Image" else (int(first_image_in.split('/')[-1].split('_')[2])// frame_rate)
        end_time = 0 if first_image_out == "No Image" else (int(first_image_out.split('/')[-1].split('_')[2])// frame_rate)


        row = {
            'ImageIn': _image_formatter(first_image_in) if in_folder_path else "No Image",
            'ID In': in_folder if in_folder is not None else "N/A",
            'Start' : seconds_to_time(start_time),
            'ImageOut': _image_formatter(first_image_out) if out_folder_path else "No Image",
            'ID Out': out_folder if out_folder is not None else "N/A",
            'End' : seconds_to_time(end_time),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert DataFrame to HTML and write to file
    html_content = df.to_html(escape=False, index=False)
    with open(filename, 'w') as file:
        file.write(html_content)


