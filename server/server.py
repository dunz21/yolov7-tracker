from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import matplotlib
import numpy as np
import os
import os
import sqlite3
from flask import g
import torch
from mini_models.re_ranking import process_re_ranking
from utils.tools import number_to_letters, seconds_to_time
    
matplotlib.use('Agg')  # Use a non-GUI backend

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/*": {"origins": "*"}})

SERVER_IP = '127.0.0.1'
# SERVER_IP = '192.168.1.87'
# SERVER_IP = '181.160.238.200'
SERVER_FOLDER_BASE_PATH = '/server-images/'
PORT = 3001
FRAME_RATE = 15

BASE_FOLDER = '/data'

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

# image['img_path'] = f"http://{SERVER_IP}:{PORT}{SERVER_FOLDER_BASE_PATH}{id}/{image['img_name']}"
@app.before_request
def before_request_func():
    project_path = request.args.get('project_path')  # Attempt to get a query parameter
    if project_path:
        projects = get_projects_available(BASE_FOLDER)
        project_data = projects.get(project_path, "Project not found")
        g.path_to_images = f"{project_path}/{project_data[0]}"
        g.path_to_db = f"{BASE_FOLDER}{project_path}/{project_data[1]}"

# @app.route(f'{SERVER_FOLDER_BASE_PATH}<path:base_folder>/<path:filename>')
# def serve_image(base_folder, filename):
#     # Define a list of allowed base folders for security
#     # allowed_base_folders = {
#     #     'folder1': '/path/to/folder1',
#     #     'folder2': '/path/to/folder2',
#     #     # Add more mappings as necessary
#     # }
    
#     # # Sanitize and validate the base_folder parameter
#     # if base_folder not in allowed_base_folders:
#     #     abort(404)  # Not found or not allowed
    
#     # # Get the absolute path to the allowed base folder
#     # abs_base_folder = allowed_base_folders[base_folder]
    
#     # # Optional: Further sanitize the filename to prevent path traversal
#     # filename = os.path.basename(filename)
    
#     return send_from_directory(f"{BASE_FOLDER}{base_folder}", filename)



### MODEL SELECTION LABELER

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
        # print(g.project_data)
        # project_path = request.args.get('project_path')
        # data = get_project_detail(project_path)
        # print(f"{BASE_FOLDER}{project_path}/{data[1]}")
        # db = get_db_connection(db_name=f"{BASE_FOLDER}{project_path}/{data[1]}")
        # project_path = f"{project_path}/{data[0]}"
        
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
            image['img_path'] = f"http://{SERVER_IP}:{PORT}{SERVER_FOLDER_BASE_PATH}{base_path_img}/{id}/{image['img_name']}"
            image['time'] = seconds_to_time(int(image['frame_number'] // FRAME_RATE))
            image['direction'] = image['img_name'].split('_')[3]
            
            
        cursor.execute("SELECT count(*) AS count FROM bbox_img_selection WHERE id = ? AND img_name IS NOT NULL", (id,))
        numberOfImages = cursor.fetchone()['count']
            
        return jsonify({'uniqueIds': sorted_unique_ids_list, 'images': images_data, 'numberOfImages': numberOfImages})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        SELECT * FROM features WHERE (id IN ({placeholders}) AND direction = 'Out')
            OR
            (direction = 'In' AND id < (SELECT MAX(id) FROM features WHERE id IN ({placeholders}) AND direction = 'Out'))
        """
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
                'image_path': f"http://{SERVER_IP}:{PORT}{SERVER_FOLDER_BASE_PATH}{base_path_img}/{img_name.split('_')[1]}/{img_name}.png",
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
        cursor.execute('''
            SELECT distinct f.id FROM features f
            WHERE NOT EXISTS (
                SELECT 1 FROM reranking_matches rm WHERE f.id = rm.id_out
            ) and f.direction='Out'
        ''')
    else:
        cursor.execute('SELECT DISTINCT id FROM features WHERE direction = "Out"')        

    rows = cursor.fetchall()
    ids_out = [row['id'] for row in rows]
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
    
    cursor.execute("SELECT count(DISTINCT f.id) as total FROM features f WHERE NOT EXISTS (SELECT 1 FROM reranking_matches rm WHERE f.id = rm.id_out) and f.direction='Out'")
    missing_matches_row = cursor.fetchone()
    missing_matches = missing_matches_row['total'] if missing_matches_row else 0
    

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
        'total_matches': total_matches,
        'missing_matches': missing_matches,
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
