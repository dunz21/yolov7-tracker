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


matplotlib.use('Agg')  # Use a non-GUI backend

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/*": {"origins": "*"}})

# SERVER_IP = '127.0.0.1'
SERVER_IP = '181.160.252.67'
PORT = 3001
FRAME_RATE = 15
FOLDER_PATH_IMGS = '/home/diego/Documents/yolov7-tracker/imgs_conce/'
SERVER_FOLDER_BASE_PATH = '/server-images/'
BASE_FOLDER_NAME = 'logs'
VIDEO_PATH = '/home/diego/Documents/Footage/CONCEPCION_CH1.mp4'  # Your video file path
BBOX_CSV = 'conce_bbox.csv'
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)

