from flask import Flask, request, jsonify, send_from_directory, abort
import json
import traceback
from flask_cors import CORS
import matplotlib
import numpy as np
import os
import os
import sqlite3
from flask import g
import torch
from mini_models.re_ranking import process_re_ranking
from utils.tools import number_to_letters
from utils.time import seconds_to_time
import pandas as pd
import datetime    
import pymysql
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64
from utils.data_analysis import get_overlap_undefined,get_direction_info
from flask_caching import Cache
from utils.types import Direction
# Configure cache
cache_config = {
    "DEBUG": True,           # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
matplotlib.use('Agg')  # Use a non-GUI backend

app = Flask(__name__)
app.config.from_mapping(cache_config)
cache = Cache(app)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/*": {"origins": "*"}})

SERVER_IP = '127.0.0.1'
# SERVER_IP = '181.160.228.136'
# SERVER_IP = '181.160.238.200'
SERVER_FOLDER_BASE_PATH = '/server-images/'
PORT = 3002
FRAME_RATE = 15
HOST, ADMIN, PASS, DB =  'mivo-db.cj2ucwgierrs.us-east-1.rds.amazonaws.com', 'admin', '58#64KDashz^bLrqTG2', 'mivo'
BASE_FOLDER = '/home/diego/Documents/yolov7-tracker/runs/detect/'

def get_db_connection():
    print(g.path_to_db)
    conn = sqlite3.connect(g.path_to_db)
    conn.row_factory = sqlite3.Row
    return conn
def get_base_folder_path():
    path = g.path_to_images
    return path


@app.route(f"{SERVER_FOLDER_BASE_PATH}<path:filename>")
def serve_image(filename):
    return send_from_directory(BASE_FOLDER, filename)

# image['img_path'] = f"{SERVER_FOLDER_BASE_PATH}{id}/{image['img_name']}"
@app.before_request
def before_request_func():
    project_path = request.args.get('project_path')  # Attempt to get a query parameter
    if project_path:
        projects = get_projects_available(BASE_FOLDER)
        project_data = projects.get(project_path, "Project not found")
        g.path_to_images = f"{project_path}/{project_data[0]}"
        g.path_to_db = f"{BASE_FOLDER}{project_path}/{project_data[1]}"

def get_projects_available(base_path):
    projects = {}

    if os.path.exists(base_path) and os.path.isdir(base_path):
        # Sort the items in the base directory in ascending order before iterating
        for item in sorted(os.listdir(base_path)):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Also sort files and folders inside each project folder
                files_and_folders = sorted(os.listdir(item_path))
                selected_items = []
                
                # Look for the first folder
                for sub_item in files_and_folders:
                    if os.path.isdir(os.path.join(item_path, sub_item)):
                        selected_items.append(sub_item)
                        break
                
                # Look for the first .db file
                for sub_item in files_and_folders:
                    if sub_item.endswith('.db'):
                        selected_items.append(sub_item)
                        break

                if selected_items:
                    projects[item] = selected_items
    
    return projects

@app.route('/api/select_project', methods=['GET'])
def select_project():
    projects = get_projects_available(BASE_FOLDER)
    return jsonify(projects)

@app.route('/api/data-images/', defaults={'id': None})
@app.route('/api/data-images/<id>')
def data_images(id): 
    try:
        
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("SELECT DISTINCT id FROM bbox_img_selection")
        unique_ids = [row['id'] for row in cursor.fetchall()]
        sorted_unique_ids_list = sorted(unique_ids, key=int)

        if id is None:
            id = unique_ids[0]
        
        

        cursor.execute("SELECT img_name, k_fold, label_img, id, area, overlap, conf_score,frame_number,selected_image FROM bbox_img_selection WHERE id = ? AND img_name != '' AND (k_fold IS NOT NULL OR k_fold_selection IS NOT NULL)", (id,))
        images_data = [dict(row) for row in cursor.fetchall()]
        base_path_img = get_base_folder_path()
        for image in images_data:
            image['img_path'] = f"{SERVER_FOLDER_BASE_PATH}{base_path_img}/{id}/{image['img_name']}"
            image['time'] = seconds_to_time(int(image['frame_number'] // FRAME_RATE))
            image['direction'] = image['img_name'].split('_')[3]
        
        
        
        video = "/home/diego/Documents/Footage/CONCEPCION_CH1.mp4"
        df = pd.read_csv('/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_17_conce_bytetrack/conce_bbox.csv')
        ID_TO_TRACK = int(id)
        # time_stamp = '00:30:09'  # The time stamp where you want to capture the image
        time_stamp = images_data[0]['time']
        hours, minutes, seconds = map(int, time_stamp.split(':')) 
        total_seconds = hours * 3600 + minutes * 60 + seconds
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError("Cannot open the video file")
        cap.set(cv2.CAP_PROP_POS_MSEC, total_seconds * 1000)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"Cannot read the frame at {time_stamp}")
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])
        rows = df.loc[df['id'] == ID_TO_TRACK]
        centroids = [(x, y) for x, y in zip(rows['centroid_x'], rows['centroid_y'])]
        centroid_middle = [((x1 + x2) // 2,y2) for x1, y1,x2,y2 in zip(rows['x1'], rows['y1'], rows['x2'], rows['y2'])]
        norm = plt.Normalize(0, len(centroid_middle)-1)
        previous_centroid = None  # Initialize previous centroid
        for i, centroid in enumerate(centroid_middle):
            color = cmap(norm(i))
            color = tuple([int(x*255) for x in color[0:3]][::-1])
            if previous_centroid is not None:
                cv2.arrowedLine(frame, previous_centroid, centroid, color, 2, tipLength=0.5)
            previous_centroid = centroid  # Update the previous centroid

        cap.release()
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        
        
            
        cursor.execute("SELECT count(*) AS count FROM bbox_img_selection WHERE id = ? AND img_name IS NOT NULL", (id,))
        numberOfImages = cursor.fetchone()['count']
            
        return jsonify({'uniqueIds': sorted_unique_ids_list, 'images': images_data, 'numberOfImages': numberOfImages, 'img_direction': img_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def create_plot(df, primary_id, intersecting_id_colors=['green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']):
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))  # You can adjust the size as needed

    # Filter the dataframe for the primary ID and sort it
    df_primary_id = df[df['id'] == primary_id].sort_values(by='frame_number')

    # Scatter plot for the primary ID with different colors for positive and negative distances
    positive_distance = df_primary_id['distance_to_center'] > 0
    ax.scatter(df_primary_id[positive_distance]['frame_number'], df_primary_id[positive_distance]['distance_to_center'],
               c='blue', s=10, alpha=0.6, label=f'ID {primary_id} In')
    ax.scatter(df_primary_id[~positive_distance]['frame_number'], df_primary_id[~positive_distance]['distance_to_center'],
               c='red', s=10, alpha=0.6, label=f'ID {primary_id} Out')

    # Calculate the time frame start and end
    timeframe_start = df_primary_id['frame_number'].min()
    timeframe_end = df_primary_id['frame_number'].max()

    # Identify other intersecting IDs within this timeframe
    intersecting_ids = df[(df['frame_number'] >= timeframe_start) & 
                          (df['frame_number'] <= timeframe_end) & 
                          (df['id'] != primary_id)]['id'].unique()

    # Plot data for each intersecting ID within the timeframe
    for idx, other_id in enumerate(intersecting_ids):
        df_other_id = df[(df['id'] == other_id) & 
                         (df['frame_number'] >= timeframe_start) & 
                         (df['frame_number'] <= timeframe_end)]
        color = intersecting_id_colors[idx % len(intersecting_id_colors)]
        ax.scatter(df_other_id['frame_number'], df_other_id['distance_to_center'], 
                   c=color, s=20, edgecolor='k', label=f'ID {other_id}')
    
    # Plot formatting
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(timeframe_start - 10, timeframe_end + 10)  # A little space around the edges
    ax.set_ylim(df_primary_id['distance_to_center'].min() - 10, df_primary_id['distance_to_center'].max() + 10)

    # Construct the title to include the ID and its timeframe
    duration = timeframe_end - timeframe_start
    title_text = f'ID {primary_id}: #{duration} '
    ax.set_title(title_text, color='red')

    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Distance to Center')
    ax.legend(loc='upper left', fontsize='small')
    
    
    # Save the figure
    base_path_img = get_base_folder_path()
    plot_path = f"{BASE_FOLDER}{base_path_img}/chart.png"
    
    plt.savefig(plot_path)
    plt.close(fig)  # Close the figure to free memory
    return f"{SERVER_FOLDER_BASE_PATH}{base_path_img}/chart.png"
    
@app.route('/api/analysis-data-images', methods=['POST'])
def analysis_data_images():
    try:
        db = get_db_connection()
        cursor = db.cursor()
        
        data = request.get_json()
        generate_report = data['generate']
        filterCounts = data['filterCounts']
        
        if generate_report:
            bbox = pd.read_sql('SELECT * FROM bbox_raw', db)
            bbox['direction'] = bbox.apply(lambda row: ('undefined' if row['img_name'].split('_')[3] == 'None' else  row['img_name'].split('_')[3]) if row['img_name'] is not None else None, axis=1)
            bbox['time_sec'] = bbox.apply(lambda row: int(row['frame_number']) // FRAME_RATE, axis=1)
            bbox['time_video'] = pd.to_datetime(bbox['time_sec'], unit='s').dt.time
            
            overlap_results = get_overlap_undefined(bbox, 0,['undefined'])
            overlap_results.to_sql('overlap_results', db, if_exists='replace', index=False)
            return jsonify({'success': True})
        
        total_counts = pd.read_sql('SELECT * FROM overlap_results', db)
        total_counts_dict = total_counts['count'].value_counts().to_dict()
        # Start with the base query
        query = 'SELECT * FROM overlap_results'
        # Check if filterCounts has items, and if so, modify the query to include a WHERE clause
        if len(filterCounts) > 0:
            counts_placeholder = ', '.join('?' for _ in filterCounts)  # Create a placeholder for each count
            query += f' WHERE count IN ({counts_placeholder})'

        # Now, execute the query. If filterCounts has items, pass them as parameters to the query.
        if len(filterCounts) > 0:
            overlap_results = pd.read_sql(query, db, params=filterCounts)
        else:
            overlap_results = pd.read_sql(query, db)
            
        sorted_unique_ids_list = sorted([int(id) for id in overlap_results['id'].unique()], key=int)

        # Get the id from the data
        id = data.get('currentId')
        # If id is None or not in the sorted_unique_ids_list, set it to the first item of sorted_unique_ids_list
        if id is None or id not in sorted_unique_ids_list:
            id = sorted_unique_ids_list[0]
        
        
        overlap_results_only_id = overlap_results[overlap_results['id'] == int(id)]
        
        ids_overlap = list(overlap_results_only_id['id_overlap'])
        ids_overlap.append(int(id))  # Append the integer version of id

        placeholders = ', '.join(['?'] * len(ids_overlap))

        query = """
            SELECT img_name, k_fold, label_img, id, area, overlap, conf_score, frame_number, selected_image FROM bbox_img_selection 
            WHERE id IN ({}) AND img_name != '' AND (k_fold IS NOT NULL OR k_fold_selection IS NOT NULL)
            """.format(placeholders)

        # Execute the query with the ids_overlap tuple as parameter
        cursor.execute(query, ids_overlap)
        images_data = [dict(row) for row in cursor.fetchall()]
        base_path_img = get_base_folder_path()
        for image in images_data:
            image['direction'] = image['img_name'].split('_')[3]
            image['img_path'] = f"{SERVER_FOLDER_BASE_PATH}{base_path_img}/{image['id']}/{image['img_name']}"
            image['time'] = seconds_to_time(int(image['frame_number'] // FRAME_RATE))

        
        overlap_results_only_id_dict = overlap_results_only_id.to_dict(orient='records')
        
        query = 'SELECT id, distance_to_center, frame_number FROM bbox_raw WHERE id IN ({})'.format(', '.join(['?']*len(ids_overlap)))
        bbox2 = pd.read_sql(query, db, params=ids_overlap)

        plot_path = create_plot(bbox2,int(id))

        for overlap in overlap_results_only_id_dict:
            # Convert datetime.time to strings
            # if 'start_time' in overlap:
            #     overlap['start_time'] = overlap['start_time'].strftime('%H:%M:%S')
            # if 'end_time' in overlap:
            #     overlap['end_time'] = overlap['end_time'].strftime('%H:%M:%S')
            # if 'id_overlap_start_time' in overlap:
            #     overlap['id_overlap_start_time'] = overlap['id_overlap_start_time'].strftime('%H:%M:%S')
            # if 'id_overlap_end_time' in overlap:
            #     overlap['id_overlap_end_time'] = overlap['id_overlap_end_time'].strftime('%H:%M:%S')
                
            overlap['bboxes'] = bbox2[bbox2['id'] == overlap['id_overlap']].to_dict(orient='records')
            overlap['images'] = [image for image in images_data if image['id'] == overlap['id_overlap']]
            overlap['images_source'] = [image for image in images_data if image['id'] == overlap['id']]

        return jsonify({'uniqueIds': sorted_unique_ids_list, 'overlapResults' : overlap_results_only_id_dict, 'plotPath': plot_path,'counts' : total_counts_dict })
    except Exception as e:
        # Capture the traceback
        tb = traceback.format_exc()  # This contains the entire traceback information
        error_message = str(e)  # Convert the exception message to a string
        error_type = type(e).__name__  # Get the type of the exception
        
        # Log the detailed traceback
        print(tb)
        
        # WARNING: Only include the traceback in the response for debugging.
        # Remove it in production to avoid security risks.
        return jsonify({
            'error': error_message,
            'error_type': error_type,
            'traceback': tb
        }), 500

@app.route('/api/update-label-img', methods=['POST'])
def update_label():
    req_data = request.get_json()
    img_name = req_data['img_name']
    new_label = req_data['new_label']
    
    try:
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("UPDATE bbox_img_selection SET label_img = ? WHERE img_name = ?", (new_label, img_name))
        db.commit()
        
        if cursor.rowcount == 0:
            return jsonify({'error': 'Image not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': 'Error updating the database', 'details': str(e)}), 500
    
@app.route('/api/ids')
def ids():
    try:
        db = get_db_connection()
        db.row_factory = sqlite3.Row

        cursor = db.cursor()

        # Fetch unique IDs for 'in' direction
        cursor.execute(f"SELECT DISTINCT id FROM features WHERE direction = '{Direction.In.value}'")
        ids_in = [row['id'] for row in cursor.fetchall()]
        ids_in_sorted = sorted(ids_in, key=int)

        # Fetch unique IDs for 'out' direction
        cursor.execute(f"SELECT DISTINCT id FROM features WHERE direction = '{Direction.Out.value}'")
        ids_out = [row['id'] for row in cursor.fetchall()]
        ids_out_sorted = sorted(ids_out, key=int)

        return jsonify({'list_in': ids_in_sorted, 'list_out': ids_out_sorted})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/get_list_ids_out', methods=['GET'])
def get_list_ids_out():
    filter_param = request.args.get('filter', default=None, type=str)
    
    db = get_db_connection()
    cursor = db.cursor()
    if filter_param == 'matches':
        # Get only the IDs of features that have a corresponding row in reranking_matches
        cursor.execute('''
            SELECT DISTINCT f.id FROM features f
            JOIN reranking_matches rm ON f.id = rm.id_out
        ''')
    elif filter_param == 'notMatches':
        # Get only the IDs of features that do not have a corresponding row in reranking_matches
        cursor.execute(f'''
            SELECT distinct f.id FROM features f
            WHERE NOT EXISTS (
                SELECT 1 FROM reranking_matches rm WHERE f.id = rm.id_out
            ) and f.direction='{Direction.Out.value}'
        ''')
    else:
        cursor.execute(f'SELECT DISTINCT id FROM features WHERE direction = "{Direction.Out.value}"')        

    rows = cursor.fetchall()
    ids_out = [row['id'] for row in rows]
    return jsonify(ids_out=ids_out), 200

@app.route('/api/re_ranking', methods=['POST'])
def trigger_re_ranking():
    data = request.json
    ids_out = data.get('ids_out', [])
    all_param = data.get('all', True)  # Get the 'all' parameter, defaulting to True
    filter_param = data.get('filter', 'all') 

    if not ids_out:
        return jsonify({'error': 'No ids_out provided'}), 400

    try:
        db = get_db_connection()
        cursor = db.cursor()
        
        
        if filter_param == 'matches':
        # Get only the IDs of features that have a corresponding row in reranking_matches
            cursor.execute('''
                SELECT DISTINCT f.id FROM features f
                JOIN reranking_matches rm ON f.id = rm.id_out
            ''')
            rows = cursor.fetchall()
            ids_out = [row['id'] for row in rows]
        elif filter_param == 'not_matches':
            # Get only the IDs of features that do not have a corresponding row in reranking_matches
            cursor.execute('''
                SELECT f.id FROM features f
                WHERE NOT EXISTS (
                    SELECT 1 FROM reranking_matches rm WHERE f.id = rm.id_out
                )
            ''')
            rows = cursor.fetchall()
            ids_out = [row['id'] for row in rows]
        
            
        
        placeholders = ', '.join(['?'] * len(ids_out))
        ids_out_twice = ids_out + ids_out

        query = f"""
        SELECT * FROM features WHERE (id IN ({placeholders}) AND direction = '{Direction.Out.value}')
            OR
            (direction = '{Direction.In.value}' AND id < (SELECT MAX(id) FROM features WHERE id IN ({placeholders}) AND direction = '{Direction.Out.value}'))
        """
        cursor.execute(query, ids_out_twice)
        rows = cursor.fetchall()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reranking_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_in INTEGER NOT NULL,
            id_out INTEGER NOT NULL UNIQUE,
            count_matches INTEGER,
            obs TEXT,
            ground_truth BOOLEAN DEFAULT 1
        )
        ''')
        
        if all_param:
            # Will filter out the IN that has the OUT
            # QUERY FOR reranking_matches and get all the data
            # filter the rows above by the following logic
            # if re renaking matches has the id_out and id_in then
            # filter all id_in except the id_in that has the id_out
            # Fetch all data from reranking_matches
            cursor.execute('SELECT * FROM reranking_matches')
            reranking_rows = cursor.fetchall()
            reranking_data = {row['id_out']: row['id_in'] for row in reranking_rows}


        # Assuming all rows have the same number of columns and the first three are id, img_name, and direction.
        ids = np.array([row['id'] for row in rows])
        img_names = np.array([row['img_name'] for row in rows])
        directions = np.array([row['direction'] for row in rows])

        # Convert feature columns to numpy array, then to tensor
        features = np.array([list(row)[3:] for row in rows], dtype=np.float32)
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        feature_tensor = feature_tensor / feature_tensor.norm(dim=1, keepdim=True)

        
        results_list, _ = process_re_ranking(
            ids, 
            img_names, 
            directions,
            feature_tensor, 
            n_images=8, 
            max_number_back_to_compare=60, 
            K1=8, 
            K2=3, 
            LAMBDA=0.1, 
            matches=reranking_data)
        
        def format_value(tuple,query):
            query_frame_number = int(query.split('_')[2])
            img_name, distance = tuple
            distance = np.round(float(distance),decimals=2) if isinstance(distance,str) and img_name != query else ''
            img_frame_number = int(img_name.split('_')[2])
            video_time = seconds_to_time((int(img_name.split('_')[2])// FRAME_RATE))
            time = seconds_to_time(max(0,(query_frame_number - img_frame_number)) // FRAME_RATE)
            base_path_img = get_base_folder_path()
            return {
                'id': f"{img_name.split('_')[1]}_{number_to_letters(img_name.split('_')[2])}",
                'image_path': f"{SERVER_FOLDER_BASE_PATH}{base_path_img}/{img_name.split('_')[1]}/{img_name}.png",
                'time': time,
                'video_time': video_time,
                'distance': distance
            }
        
        def format_row(arr):
            new_list = []
            for row in arr:
                query = row[0][0]
                new_list.append([format_value(value,query) for value in row])
            return new_list

        list_out = {}
        for id_out in results_list:
            list_out[str(id_out)] = format_row(results_list[id_out])
        
        
        return jsonify({
            'results': list_out
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/re_ranking/simple', methods=['POST'])
def trigger_re_ranking_db():


    try:
        db = get_db_connection()
        cursor = db.cursor()

        query = f"""SELECT * FROM features """
        cursor.execute(query)
        rows = cursor.fetchall()

        # Assuming all rows have the same number of columns and the first three are id, img_name, and direction.
        ids = np.array([row['id'] for row in rows])
        img_names = np.array([row['img_name'] for row in rows])
        directions = np.array([row['direction'] for row in rows])

        # Convert feature columns to numpy array, then to tensor
        features = np.array([list(row)[3:] for row in rows], dtype=np.float32)
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        feature_tensor = feature_tensor / feature_tensor.norm(dim=1, keepdim=True)
        
        results_list, posible_pair_matches = process_re_ranking(
            ids, 
            img_names, 
            directions,
            feature_tensor, 
            n_images=8, 
            max_number_back_to_compare=60, 
            K1=8, 
            K2=3, 
            LAMBDA=0.1, 
            autoeval=True)
        
        def format_value(tuple,query):
            query_frame_number = int(query.split('_')[2])
            img_name, distance = tuple
            distance = np.round(float(distance),decimals=2) if isinstance(distance,str) and img_name != query else ''
            img_frame_number = int(img_name.split('_')[2])
            video_time = seconds_to_time((int(img_name.split('_')[2])// FRAME_RATE))
            time = seconds_to_time(max(0,(query_frame_number - img_frame_number)) // FRAME_RATE)
            base_path_img = get_base_folder_path()
            return {
                'id': f"{img_name.split('_')[1]}_{number_to_letters(img_name.split('_')[2])}",
                'image_path': f"{SERVER_FOLDER_BASE_PATH}{base_path_img}/{img_name.split('_')[1]}/{img_name}.png",
                'time': time,
                'video_time': video_time,
                'distance': distance
            }
        
        def format_row(arr):
            new_list = []
            for row in arr:
                query = row[0][0]
                new_list.append([format_value(value,query) for value in row])
            return new_list
        list_out = {}
        for id_out in results_list:
            list_out[str(id_out)] = format_row(results_list[id_out])
            
    
        cursor.execute('''DROP TABLE IF EXISTS reranking''')
        cursor.execute('''DROP TABLE IF EXISTS reranking_matches''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reranking_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_in INTEGER NOT NULL,
                id_out INTEGER NOT NULL UNIQUE,
                count_matches INTEGER,
                obs TEXT,
                ground_truth BOOLEAN DEFAULT 1
            )
            ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reranking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_out TEXT,
                img_out TEXT,
                id_in TEXT,
                img_in TEXT,
                time_diff TEXT,
                video_time TEXT,
                distance TEXT,
                rank INTEGER
            )
        ''')
        
        for id_out, top_k_list in list_out.items():
            for row_gallery in top_k_list:
                rank = 1  # Initialize rank for each id_out group
                
                cursor.execute('''
                        INSERT INTO reranking (id_out, img_out, id_in, img_in, time_diff, video_time, distance, rank)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (id_out, row_gallery[0]['image_path'], None, None, None, row_gallery[0]['video_time'], None, None))
                
                for item in row_gallery[1:]:
                    img_out = row_gallery[0]['image_path']
                    id_in = item['id'].split('_')[0]
                    img_in = item['image_path']
                    time_diff = item['time']
                    video_time = item['video_time']
                    distance = str(item['distance'])  # Ensure distance is a string for consistency

                    cursor.execute('''
                        INSERT INTO reranking (id_out, img_out, id_in, img_in, time_diff, video_time, distance, rank)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (id_out, img_out, id_in, img_in, time_diff, video_time, distance, rank))

                    rank += 1  # Increment rank for each row

        db.commit()
        
        
        # Initialize an empty dictionary for the results
        predict_matches = {}
        for out_in_tuple in posible_pair_matches:
            id_out = out_in_tuple[0]
            id_in = out_in_tuple[1]
            predict_matches[id_out] = {
                'id_in': id_in
            }
            
            # Attempt to insert or update
            try:
                cursor.execute('''
                INSERT INTO reranking_matches (id_in, id_out)
                VALUES (?, ?)
                ON CONFLICT(id_out) DO UPDATE SET
                    id_in = excluded.id_in,
                    count_matches = excluded.count_matches,
                    obs = excluded.obs
                ''', (id_in, id_out))
                db.commit()
            except sqlite3.IntegrityError as e:
                return jsonify({'error': str(e)}), 500
            
        db.close() 
        return jsonify({'results': 'OK'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/re_ranking/get', methods=['POST'])
def get_re_ranking_db():
    try:
        db = get_db_connection()
        cursor = db.cursor()
        
        query = f'''SELECT * FROM reranking_matches'''
        cursor.execute(query)
        rows = cursor.fetchall()

        # Initialize an empty dictionary for the results
        matches = {}
        for row in rows:
            # Convert row to dictionary (assuming row factory is set)
            row_dict = dict(row)
            id_out = row_dict['id_out']
            # Construct the desired dictionary structure for each id_out
            matches[id_out] = {
                'count_matches': row_dict['count_matches'],
                'id_in': row_dict['id_in'],
                'obs': row_dict['obs'],
                'ground_truth': row_dict['ground_truth'],
            }
        

        # Fetch basic image data
        cursor.execute("SELECT id, img_name FROM bbox_raw WHERE img_name IS NOT NULL")
        total_values = cursor.fetchall()
        if not total_values:
            return jsonify({'error': 'No images found'}), 404

        df = pd.DataFrame(total_values, columns=['id', 'img_name'])
        df['direction'] = df['img_name'].apply(lambda x: x.split('_')[3])
        df_unique = df.drop_duplicates(subset=['id'], keep='first')
        direction_counts = df_unique['direction'].value_counts().to_dict()

        # Fetching re-ranking results from 'reranking' table
        cursor.execute("SELECT * FROM reranking ORDER BY id asc")
        reranking_data = cursor.fetchall()
        reranking_columns = [desc[0] for desc in cursor.description]
        reranking_df = pd.DataFrame(reranking_data, columns=reranking_columns)

        # Organize reranking data into the specified format
        reranking_results = {}
        current_id_out = None
        current_batch = []

        base_path_img = get_base_folder_path()
        for index, row in reranking_df.iterrows():
            # Start a new batch when rank is NULL
            if pd.isna(row['rank']):
                if current_id_out is not None and current_id_out == row['id_out']:
                    reranking_results[current_id_out].append(current_batch)
                    current_batch = []
                elif current_id_out != row['id_out']:
                    if current_id_out is not None:
                        reranking_results[current_id_out].append(current_batch)
                    current_id_out = row['id_out']
                    reranking_results[current_id_out] = []
                    current_batch = []
            img = f"{SERVER_FOLDER_BASE_PATH}{base_path_img}/{row['img_in']}" if pd.notna(row['img_in']) else f"{SERVER_FOLDER_BASE_PATH}{base_path_img}/{row['img_out']}"
            current_batch.append({
                'id_bd': row['id'],
                'id': row['id_in'] if pd.notna(row['id_in']) else row['id_out'],
                'image_path': img,
                'time': row['time_diff'],
                'video_time': row['video_time'],
                'distance': row['distance']
            })

        # Append the last batch for the last id_out
        if current_id_out and current_batch:
            reranking_results[current_id_out].append(current_batch)

        db.close()  # Always close connection

        return jsonify({
            'posible_pair_matches': matches,
            'stats': direction_counts,
            'reranking_results': reranking_results
        })

    except Exception as e:
        if db:
            db.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/reranking/ground_truth', methods=['POST'])
def update_ground_truth_re_ranking():
    # Extract data from the request
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    id_out = data.get('id_out')
    checked = data.get('checked')

    # Validate presence of all required inputs
    if id_out is None or checked is None:
        return jsonify({'error': 'Missing data: id_out and/or checked must be provided'}), 400

    # Ensure 'id_out' is an integer
    try:
        id_out = int(id_out)
    except ValueError:
        return jsonify({'error': 'Invalid data type for id_out'}), 400

    # Ensure 'checked' is a boolean
    if not isinstance(checked, bool):
        return jsonify({'error': 'Invalid data type for checked: Must be true or false'}), 400

    db = get_db_connection()
    cursor = db.cursor()

    try:
        # Update the ground_truth value based on id_out
        cursor.execute("UPDATE reranking_matches SET ground_truth = ? WHERE id_out = ?", (checked, id_out))
        db.commit()
        
        # Fetch and return the updated record
        cursor.execute("SELECT * FROM reranking_matches WHERE id_out = ?", (id_out,))
        re_ranking_value = cursor.fetchone()

        if re_ranking_value is None:
            return jsonify({'error': 'No record found with the provided id_out'}), 404

        return jsonify(dict(re_ranking_value))
    except sqlite3.IntegrityError as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()
    

@app.route('/api/reranking/match', methods=['POST'])
def insert_reranking_match():
    # Extract data from the request
    data = request.json
    id_in = data.get('id_in')
    id_out = data.get('id_out')
    count_matches = data.get('count_matches')
    obs = data.get('obs')
    # Extract the optional isSelected parameter, defaults to False if not provided
    isSelected = data.get('isSelected', False)

    if None in [id_in, id_out, count_matches, obs]:
        return jsonify({'error': 'Missing data'}), 400

    # Ensure all integer fields are indeed integers
    try:
        id_in = int(id_in)
        id_out = int(id_out)
        count_matches = int(count_matches)
    except ValueError:
        return jsonify({'error': 'Invalid data type for id_in, id_out, or count_matches'}), 400

    db = get_db_connection()
    cursor = db.cursor()

    # Check or create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reranking_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_in INTEGER NOT NULL,
        id_out INTEGER NOT NULL UNIQUE,
        count_matches INTEGER,
        obs TEXT,
        ground_truth BOOLEAN DEFAULT 1
    )
    ''')

    if isSelected:
        # If isSelected is true, delete the row with the same id_out and id_in before inserting or updating
        cursor.execute('''
        DELETE FROM reranking_matches WHERE id_out = ? AND id_in = ?
        ''', (id_out, id_in))
        db.commit()
        return jsonify({'message': 'Delete Success'}), 200
        
    
    # Attempt to insert or update
    try:
        cursor.execute('''
        INSERT INTO reranking_matches (id_in, id_out, count_matches, obs)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id_out) DO UPDATE SET
            id_in = excluded.id_in,
            count_matches = excluded.count_matches,
            obs = excluded.obs
        ''', (id_in, id_out, count_matches, obs))
        db.commit()
    except sqlite3.IntegrityError as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': 'Success'}), 200

@app.route('/api/reranking/match', methods=['GET'])
def get_reranking_matches():
    # Get ids_out from query parameter as a comma-separated string
    ids_out_str = request.args.get('ids_out', '')
    # Convert to a list of integers
    try:
        ids_out = [int(id_str) for id_str in ids_out_str.split(',') if id_str]
    except ValueError:
        return jsonify({'error': 'Invalid ids_out. Must be a list of integers.'}), 400

    if not ids_out:
        return jsonify({'error': 'No ids_out provided'}), 400

    db = get_db_connection()
    cursor = db.cursor()

    # Check and create the reranking_matches table if it does not exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reranking_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_in INTEGER NOT NULL,
        id_out INTEGER NOT NULL UNIQUE,
        count_matches INTEGER,
        obs TEXT,
        ground_truth BOOLEAN DEFAULT 1
    )
    ''')

    # Prepare placeholders for the query
    placeholders = ', '.join(['?'] * len(ids_out))
    query = f'''
    SELECT * FROM reranking_matches WHERE id_out IN ({placeholders})
    '''
    cursor.execute(query, ids_out)
    rows = cursor.fetchall()

    # Initialize an empty dictionary for the results
    matches = {}
    for row in rows:
        # Convert row to dictionary (assuming row factory is set)
        row_dict = dict(row)
        id_out = row_dict['id_out']
        # Construct the desired dictionary structure for each id_out
        matches[id_out] = {
            'count_matches': row_dict['count_matches'],
            'id_in': row_dict['id_in'],
            'obs': row_dict['obs']
        }

    db.close()  # Don't forget to close the database connection
    return jsonify(matches), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    db = get_db_connection()
    cursor = db.cursor()

    # Query for counting unique 'OUT' and 'IN' in the features table
    cursor.execute("SELECT direction, COUNT(DISTINCT id) AS total FROM features GROUP BY direction")
    direction_counts = cursor.fetchall()
    
    cursor.execute(f"SELECT count(DISTINCT f.id) as total FROM features f WHERE NOT EXISTS (SELECT 1 FROM reranking_matches rm WHERE f.id = rm.id_out) and f.direction='{Direction.Out.value}'")
    missing_matches_row = cursor.fetchone()
    missing_matches = missing_matches_row['total'] if missing_matches_row else 0
    

    # Initialize counts
    out_count = 0
    in_count = 0

    # Iterate through the results and assign the counts
    for row in direction_counts:
        if row['direction'] == Direction.Out.value:
            out_count = row['total']
        elif row['direction'] == Direction.In.value:
            in_count = row['total']

    # Query for total number of rows in the reranking_matches table
    cursor.execute("SELECT COUNT(*) AS total_matches FROM reranking_matches")
    total_matches_row = cursor.fetchone()
    total_matches = total_matches_row['total_matches'] if total_matches_row else 0
    

    # Structuring the response
    response = {
        'in': in_count,
        'out': out_count,
        'total_matches': total_matches,
        'total_auto': 0,
        'missing_matches': missing_matches,
    }

    return jsonify(response)

##### Next Mivo #####


def modify_count(data, min_count=20, max_count=70):
    import random
    for item in data:
        item['count'] = random.randint(min_count, max_count)
    return data

@app.route('/api/process_data', methods=['GET'])
def process_data():
    direction_param = request.args.get('direction', Direction.In.value)  # Get direction parameter, default 'In'
    
    # Connect to the SQLite database
    db = get_db_connection()
    # Load the 'bbox_raw' table into a DataFrame
    df = pd.read_sql_query("SELECT * FROM bbox_raw WHERE img_name IS NOT NULL", db)
    
    # Ensure img_name has a value
    df = df.dropna(subset=['img_name'])
    
    # Drop duplicates based on 'id' to keep only one row per id
    df = df.drop_duplicates(subset=['id'])
    
    # Add 'direction' column by splitting 'img_name' and extracting the fourth element
    df['direction'] = df['img_name'].apply(lambda x: x.split('_')[3])
    
    # Filter dataframe for rows where direction is either 'In' or 'Out'
    df = df[df['direction'].isin([Direction.In.value, Direction.Out.value])]
    
    # Add 'time' column calculated from 'frame_number' divided by 15, rounded to hours
    df['time'] = df['frame_number'].apply(lambda x: seconds_to_time((x / 15) + (60 * 60 * 8)))
    
    # Filter by direction based on input parameter
    df = df[df['direction'] == direction_param]
    
    # Convert 'time' to datetime to facilitate grouping by hour
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    
    # Get the full hour range from min to max
    hours_range = range(df['hour'].min(), df['hour'].max() + 1)
    
    # Group data by hour and count the occurrences
    grouped_data = df.groupby('hour').size().reindex(hours_range, fill_value=0).reset_index(name='count')
    grouped_data['time'] = grouped_data['hour'].apply(lambda x: f"{x:02}:00")
    grouped_data = grouped_data[['count', 'time']]
    
    # Close the database connection
    db.close()
    connection = pymysql.connect(host=HOST, user=ADMIN, password=PASS, database=DB)
    
    data = grouped_data.to_dict(orient='records')
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-01",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-02",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-03",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-04",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-05",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-06",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-07",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-08",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-09",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-10",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-11",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-12",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 1, "2024-04-13",connection)
    
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-01",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-02",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-03",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-04",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-05",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-06",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-07",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-08",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-09",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-10",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-11",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-12",connection)
    # data = modify_count(data,20,80)
    # save_visits(data, 2, "2024-04-13",connection)
    # data = modify_count(data,20,80)
    # connection.close()

    return jsonify(data)

def save_visits(data, store_id, date, connection):
    try:
        with connection.cursor() as cursor:
            # SQL statement to insert data
            sql = "INSERT INTO visits (`count`, `time`, `store_id`, `date`) VALUES (%s, %s, %s, %s)"
            # Prepare data for insertion
            for item in data:
                cursor.execute(sql, (item['count'], item['time'], store_id, date))
        connection.commit()
    finally:
        print(date)

@app.route('/api/visits', methods=['GET'])
def get_visits():
    store = request.args.get('store')
    date = request.args.get('date')
    
    if not store or not date:
        return jsonify({"error": "Missing 'store' or 'date' parameter"}), 400 

    connection = pymysql.connect(host=HOST, user=ADMIN, password=PASS, database=DB)
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT id, count, date, time, store_id FROM visits WHERE store_id = %s AND date = %s"
            cursor.execute(sql, (store, date))
            stores = cursor.fetchall()
            for store in stores:  # Ensure that timedelta objects are processed
                if isinstance(store['time'], datetime.timedelta):
                    store['time'] = str(store['time'])
            return jsonify(stores)
    except pymysql.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    finally:
        connection.close()
        
@app.route('/api/stores', methods=['GET'])
def get_store_data():
    
    
    connection = pymysql.connect(host=HOST, user=ADMIN, password=PASS, database=DB)
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:  # Key change here
            sql = "SELECT id, name, full_name, clients_id FROM stores"
            cursor.execute(sql)
            stores = cursor.fetchall()
            return jsonify(stores) 
    finally:
        connection.close()
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)



