from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
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

matplotlib.use('Agg')  # Use a non-GUI backend

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

PORT = 3001
FOLDER_PATH_IMGS = '/home/diego/Documents/yolov7-tracker/imgs_conce/'
SERVER_FOLDER_BASE_PATH = '/server-images/'
DATA_FILE_JSON = '/home/diego/Documents/yolov7-tracker/logs/selected_images.json'
BASE_FOLDER_NAME = 'logs_e'
VIDEO_PATH = '/home/diego/Documents/Footage/CONCEPCION_CH1.mp4'  # Your video file path


BBOX_CSV = 'conce_bbox.csv_alternative.csv'
BBOX_CSV = os.path.join(BASE_FOLDER_NAME, BBOX_CSV)


@app.route(f"{SERVER_FOLDER_BASE_PATH}<path:filename>")
def serve_image(filename):
    return send_from_directory(FOLDER_PATH_IMGS, filename)

@app.route('/api/data-images/<id>', defaults={'id': None})
@app.route('/api/data-images/', defaults={'id': None})
def data_images(id):
    try:
        with open(DATA_FILE_JSON, 'r') as file:
            images = json.load(file)

        unique_ids = list(set([image['id'] for image in images]))
        filtered_images = images

        for image in filtered_images:
            image['img'] = f"http://localhost:{PORT}{image['img'].replace(FOLDER_PATH_IMGS, SERVER_FOLDER_BASE_PATH)}"

        if id is not None:
            filtered_images = [image for image in filtered_images if str(image['id']) == id]

        return jsonify({'uniqueIds': unique_ids, 'images': filtered_images})
    except IOError:
        return jsonify({'error': 'Error reading data file'}), 500

@app.route('/api/update-rate', methods=['POST'])
def update_rate():
    req_data = request.get_json()
    img = req_data['img']
    new_rate = req_data['newRate']

    try:
        with open(DATA_FILE_JSON, 'r') as file:
            images = json.load(file)

        for image in images:
            if image['img'].replace(f"http://localhost:{PORT}{SERVER_FOLDER_BASE_PATH}", FOLDER_PATH_IMGS) == img:
                image['rate'] = str(new_rate)

        with open(DATA_FILE_JSON, 'w') as file:
            json.dump(images, file, indent=2)

        return jsonify({'success': True})
    except IOError:
        return jsonify({'error': 'Error reading or updating data file'}), 500
    

@app.route('/api/in-out-images', methods=['GET'])
def in_out_images():
    id_param = request.args.get('id', default=None, type=int)
    
    time_stamp = '00:00:01'  # Timestamp for the frame
    
    df = pd.read_csv(BBOX_CSV)

    if id_param is None:
        unique_ids_list = df['id'].unique().tolist()
        return jsonify({'uniqueIds': unique_ids_list})

    # Finding the row for the specific ID
    rows = df.loc[df['id'] == id_param]
    if rows.empty:
        return jsonify({'error': f'ID {id_param} not found in the dataset'}), 404

    # Check if the 'label_direction' column exists and get its value if it does
    direction = None
    if 'label_direction' in df.columns:
        direction_value = rows.iloc[0]['label_direction'] if not rows.empty else np.nan
        direction = None if pd.isna(direction_value) else direction_value

    
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
    centroids = [(x, y) for x, y in zip(rows['centroid_x'], rows['centroid_y'])]
    norm = plt.Normalize(0, len(centroids)-1)
    
    # Drawing circles with gradient colors
    for i, centroid in enumerate(centroids):
        color = cmap(norm(i))
        color = tuple([int(x*255) for x in color[:3]][::-1])  # Convert from RGB to BGR
        cv2.circle(frame, tuple(map(int, centroid)), 5, color, -1)
    
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
    
    return jsonify({'image': img_base64, 'id': id_param, 'direction': direction})


@app.route('/api/in-out-images/<int:id>', methods=['POST'])
def update_direction(id):
    data = request.get_json()
    direction = data.get('direction')

    if direction is None:
        return jsonify({'error': 'Missing direction in request'}), 400

    # Load the CSV file
    df = pd.read_csv(BBOX_CSV)

    # Check if the ID exists
    if id not in df['id'].values:
        return jsonify({'error': 'ID not found'}), 404

    # Update the direction for the given ID
    df.loc[df['id'] == id, 'label_direction'] = direction

    # Save the updated DataFrame back to CSV
    df.to_csv(BBOX_CSV, index=False)

    return jsonify({'message': 'Direction updated successfully', 'id': id, 'direction': direction})

if __name__ == '__main__':
    app.run(port=PORT, debug=True)
