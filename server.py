from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import pandas as pd
import matplotlib
import cv2
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os
from utils.solider import seconds_to_time
import sqlite3
from flask import g
import torch
from mini_models.re_ranking import process_re_ranking
import datetime
    
    
matplotlib.use('Agg')  # Use a non-GUI backend

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/*": {"origins": "*"}})

# SERVER_IP = '127.0.0.1'
SERVER_IP = '181.161.112.110'
PORT = 3001
FRAME_RATE = 15
FOLDER_PATH_IMGS = '/home/diego/Documents/yolov7-tracker/imgs_santos_dumont_top4/'
VIDEO_PATH = '/home/diego/Documents/Footage/SANTOS LAN_ch6.mp4'  # Your video file path
BBOX_CSV = 'conce_bbox.csv'
SERVER_FOLDER_BASE_PATH = '/server-images/'
BASE_FOLDER_NAME = 'logs'
BBOX_CSV = os.path.join(BASE_FOLDER_NAME, BBOX_CSV)

#### DATABASE #####
DATABASE = f'{BASE_FOLDER_NAME}/bbox_data.db'
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

### END DATABASE ###

@app.route(f"{SERVER_FOLDER_BASE_PATH}<path:filename>")
def serve_image(filename):
    return send_from_directory(FOLDER_PATH_IMGS, filename)

### MODEL SELECTION LABELER

@app.route('/api/data-images/', defaults={'id': None})
@app.route('/api/data-images/<id>')
def data_images(id): 
    try:
        db = get_db()
        db.row_factory = sqlite3.Row  # Access columns by name
        
        cursor = db.cursor()
        cursor.execute("SELECT DISTINCT id FROM bbox_data")
        unique_ids = [row['id'] for row in cursor.fetchall()]
        sorted_unique_ids_list = sorted(unique_ids, key=int)

        if id is None:
            id = unique_ids[0]
        
        cursor.execute("SELECT img_name, k_fold, label_img, id, area, overlap, conf_score FROM bbox_data WHERE id = ? AND img_name != '' AND k_fold IS NOT NULL", (id,))
        images_data = [dict(row) for row in cursor.fetchall()]

        for image in images_data:
            image['img_path'] = f"http://{SERVER_IP}:{PORT}{SERVER_FOLDER_BASE_PATH}{id}/{image['img_name']}"
            image['label_img'] = None if image['label_img'] is None else image['label_img']
            
        return jsonify({'uniqueIds': sorted_unique_ids_list, 'images': images_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-label-img', methods=['POST'])
def update_label():
    req_data = request.get_json()
    img_name = req_data['img_name']
    new_label = req_data['new_label']
    
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("UPDATE bbox_data SET label_img = ? WHERE img_name = ?", (new_label, img_name))
        db.commit()
        
        if cursor.rowcount == 0:
            return jsonify({'error': 'Image not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': 'Error updating the database', 'details': str(e)}), 500
    


### IN OUT IMAGES and BAD IMAGES LABELER

@app.route('/api/in-out-images', methods=['GET'])
def in_out_images():
    db = get_db()
    db.row_factory = sqlite3.Row  # Access columns by name
    
    id_param = request.args.get('id', default=None, type=int)
    time_stamp = '00:00:01'  # Timestamp for the frame
    
    # Fetch unique IDs
    cursor = db.execute('SELECT DISTINCT id FROM bbox_data')
    unique_ids_list = [row['id'] for row in cursor.fetchall()]
    sorted_unique_ids_list = sorted(unique_ids_list, key=int)


    if id_param is None:
        return jsonify({'uniqueIds': sorted_unique_ids_list})
    
    # Finding the rows for the specific ID
    cursor = db.execute('SELECT * FROM bbox_data WHERE id = ?', (id_param,))
    rows = cursor.fetchall()
    if not rows:
        return jsonify({'error': f'ID {id_param} not found in the dataset'}), 404
    

    # Check if the 'label_direction' column exists and get its value if it does
    direction = None
    # Assume 'label_direction' column exists; adjust as necessary
    if rows[0]['label_direction'] is not None:
        direction = rows[0]['label_direction']

    
    hours, minutes, seconds = map(int, time_stamp.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open the video file'}), 500
    cap.set(cv2.CAP_PROP_POS_MSEC, total_seconds * 1000)
    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': f'Cannot read the frame at {time_stamp}'}), 500
    
    # Get video frame dimensions
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Colormap setup
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])
    norm = plt.Normalize(0, len(rows)-1)
    images_data = []
    previous_centroid = None  # Initialize previous centroid

    for i,row in enumerate(rows):
        # Calculate centroid_bottom_x and centroid_bottom_y for each row
        centroid_bottom_x = (row['x1'] + row['x2']) // 2
        centroid_bottom_y = row['y2']
        centroid = (centroid_bottom_x, centroid_bottom_y)

        color = cmap(norm(i))
        color = tuple([int(x*255) for x in color[:3]][::-1])  # Convert from RGB to BGR
        # cv2.circle(frame, tuple(map(int, centroid)), 5, color, -1)
        if previous_centroid is not None:
            cv2.arrowedLine(frame, previous_centroid, centroid, color, 2, tipLength=0.5)
        
        previous_centroid = centroid  # Update the previous centroid

        if row['img_name']:
            images_data.append(
                {
                    'img_name': row['img_name'],
                    'img_path': f"http://{SERVER_IP}:{PORT}{SERVER_FOLDER_BASE_PATH}{id_param}/{row['img_name']}",
                }
            )

    
    cap.release()
    
    # Convert the frame to a format suitable for JSON response
    frame_with_figure_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set figure size based on video resolution
    dpi = 100.0
    figsize = (video_width / dpi, video_height / dpi)  # Figure size in inches
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(frame_with_figure_rgb)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Convert figure to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    time_video = seconds_to_time(int(rows[0]['frame_number'] // FRAME_RATE))
    
    return jsonify({'image': img_base64, 'id': id_param, 'direction': direction,'images': images_data,'time_video': time_video})


@app.route('/api/in-out-images/<int:id>', methods=['POST'])
def update_direction(id):
    data = request.get_json()
    direction = data.get('direction')

    if direction is None:
        return jsonify({'error': 'Missing direction in request'}), 400

    db = get_db()
    cursor = db.cursor()

    # Check if the ID exists
    cursor.execute('SELECT * FROM bbox_data WHERE id = ?', (id,))
    if cursor.fetchone() is None:
        return jsonify({'error': 'ID not found'}), 404

    # Update the direction for the given ID
    cursor.execute('UPDATE bbox_data SET label_direction = ? WHERE id = ?', (direction, id))
    db.commit()

    if cursor.rowcount == 0:
        # No rows were updated, indicating the ID was not found
        return jsonify({'error': 'ID not found'}), 404

    return jsonify({'message': 'Direction updated successfully', 'id': id, 'direction': direction})


def get_db_connection(db_name="output/santos_dumont_solider_in-out_DB.db"):
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/ids')
def ids():
    try:
        db = get_db_connection()
        db.row_factory = sqlite3.Row

        cursor = db.cursor()

        # Fetch unique IDs for 'in' direction
        cursor.execute("SELECT DISTINCT id FROM features WHERE direction = 'In'")
        ids_in = [row['id'] for row in cursor.fetchall()]
        ids_in_sorted = sorted(ids_in, key=int)

        # Fetch unique IDs for 'out' direction
        cursor.execute("SELECT DISTINCT id FROM features WHERE direction = 'Out'")
        ids_out = [row['id'] for row in cursor.fetchall()]
        ids_out_sorted = sorted(ids_out, key=int)

        return jsonify({'list_in': ids_in_sorted, 'list_out': ids_out_sorted})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/re_ranking', methods=['POST'])
def re_ranking():
    data = request.json
    ids_out = data.get('ids_out', [])
    all_param = data.get('all', True)  # Get the 'all' parameter, defaulting to True

    if not ids_out:
        return jsonify({'error': 'No ids_out provided'}), 400

    try:
        db = get_db_connection()
        placeholders = ', '.join(['?'] * len(ids_out))
        ids_out_twice = ids_out + ids_out

        query = f"""
        SELECT * FROM features WHERE (id IN ({placeholders}) AND direction = 'Out')
            OR
            (direction = 'In' AND id < (SELECT MAX(id) FROM features WHERE id IN ({placeholders}) AND direction = 'Out'))
        """
        cursor = db.cursor()
        cursor.execute(query, ids_out_twice)
        rows = cursor.fetchall()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reranking_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_in INTEGER NOT NULL,
            id_out INTEGER NOT NULL UNIQUE,
            count_matches INTEGER NOT NULL,
            obs TEXT NOT NULL
        )
        ''')
        
        if all_param:
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

        
        results_list, _ = process_re_ranking(ids, img_names, directions,feature_tensor, n_images=5, max_number_back_to_compare=60, K1=8, K2=3, LAMBDA=0.1, matches=reranking_data)
        
        
        def seconds_to_time(seconds):
            td = datetime.timedelta(seconds=seconds)
            time = (datetime.datetime.min + td).time()
            return time.strftime("%H:%M:%S")
        def number_to_letters(num):
            mapping = {i: chr(122 - i) for i in range(10)}
            num_str = str(num)
            letter_code = ''.join(mapping[int(digit)] for digit in num_str)
            return letter_code
        def format_value(img_name,query_frame_number):
            img_frame_number = int(img_name.split('_')[2])
            video_time = seconds_to_time((int(img_name.split('_')[2])// FRAME_RATE))
            time = seconds_to_time(max(0,(query_frame_number - img_frame_number)) // FRAME_RATE)
            return {
                'id': f"{img_name.split('_')[1]}_{number_to_letters(img_name.split('_')[2])}",
                'image_path': f"http://{SERVER_IP}:{PORT}{SERVER_FOLDER_BASE_PATH}{img_name.split('_')[1]}/{img_name}.png",
                'time': time,
                'video_time': video_time
            }
        
        def format_row(arr):
            new_list = []
            for row in arr:
                query = row[0]
                query_frame_number = int(query.split('_')[2])
                new_list.append([format_value(value,query_frame_number) for value in row])
            return new_list

        list_out = {}
        for id_out in results_list:
            list_out[str(id_out)] = format_row(results_list[id_out])
        
        
        return jsonify({
            'results': list_out
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/get_list_ids_out', methods=['GET'])
def get_list_ids_out():
    db = get_db_connection()
    cursor = db.cursor()
    
    # Select distinct id values from the features table
    cursor.execute('SELECT DISTINCT id FROM features WHERE direction = "Out"')
    rows = cursor.fetchall()

    # Extract the id values from the rows
    ids_out = [row['id'] for row in rows]

    # Return the list of unique id values
    return jsonify(ids_out=ids_out), 200

@app.route('/api/reranking/match', methods=['POST'])
def reranking_match():
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
        count_matches INTEGER NOT NULL,
        obs TEXT NOT NULL
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
        count_matches INTEGER NOT NULL,
        obs TEXT NOT NULL
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

    # Initialize counts
    out_count = 0
    in_count = 0

    # Iterate through the results and assign the counts
    for row in direction_counts:
        if row['direction'] == 'Out':
            out_count = row['total']
        elif row['direction'] == 'In':
            in_count = row['total']

    # Query for total number of rows in the reranking_matches table
    cursor.execute("SELECT COUNT(*) AS total_matches FROM reranking_matches")
    total_matches_row = cursor.fetchone()
    total_matches = total_matches_row['total_matches'] if total_matches_row else 0

    # Structuring the response
    response = {
        'in': in_count,
        'out': out_count,
        'total_matches': total_matches
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)

