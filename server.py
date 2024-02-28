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


matplotlib.use('Agg')  # Use a non-GUI backend

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

PORT = 3001
FRAME_RATE = 15
FOLDER_PATH_IMGS = '/home/diego/Documents/yolov7-tracker/imgs_conce/'
SERVER_FOLDER_BASE_PATH = '/server-images/'
BASE_FOLDER_NAME = 'logs'
VIDEO_PATH = '/home/diego/Documents/Footage/CONCEPCION_CH1.mp4'  # Your video file path
BBOX_CSV = 'conce_bbox.csv'
BBOX_CSV = os.path.join(BASE_FOLDER_NAME, BBOX_CSV)

@app.route(f"{SERVER_FOLDER_BASE_PATH}<path:filename>")
def serve_image(filename):
    return send_from_directory(FOLDER_PATH_IMGS, filename)

### MODEL SELECTION LABELER

@app.route('/api/data-images/', defaults={'id': None})
@app.route('/api/data-images/<id>')
def data_images(id):
    try:
        print('id',id)
        # Read the CSV file
        df = pd.read_csv(BBOX_CSV)

        # Get unique IDs
        unique_ids = df['id'].unique().tolist()

        # Default ID to the first unique ID if None
        if id is None:
            id = unique_ids[0]
        
        # Filter rows where 'img_name' is not empty and 'k_fold' has a value
        df_filtered = df[(df['img_name'] != '') & (df['k_fold'].notna())]

        # Further filter by the requested ID
        df_filtered = df_filtered[df_filtered['id'] == int(id)]

        # Convert to a list of dictionaries for JSON response
        images_data = df_filtered[['img_name','k_fold','label_img','id']].to_dict(orient='records')

        # Modify image paths for the server
        for image in images_data:
            image['img_path'] = f"http://localhost:{PORT}{SERVER_FOLDER_BASE_PATH}{id}/{image['img_name']}"
            image['label_img'] =  None if pd.isna(image['label_img']) else image['label_img']

        return jsonify({'uniqueIds': unique_ids, 'images': images_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-label-img', methods=['POST'])
def update_label():
    req_data = request.get_json()
    img_name = req_data['img_name']
    new_label = req_data['new_label']

    try:
        # Load the CSV file
        df = pd.read_csv(BBOX_CSV)

        # Find the row with the matching image name and update the label_image column
        match = df['img_name'] == img_name
        if match.any():
            df.loc[match, 'label_image'] = new_label

            # Save the updated DataFrame back to CSV
            df.to_csv(BBOX_CSV, index=False)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Image not found'}), 404

    except Exception as e:
        return jsonify({'error': 'Error reading or updating CSV file', 'details': str(e)}), 500
    


### IN OUT IMAGES and BAD IMAGES LABELER

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


     # Prepare image data
    images_data = rows[rows['img_name'].notna()][['img_name']].to_dict(orient='records')

    for image in images_data:
        image['img_path'] = f"http://localhost:{PORT}{SERVER_FOLDER_BASE_PATH}{id_param}/{image['img_name']}"

    time_video = seconds_to_time(int(rows.iloc[0]['frame_number'] // FRAME_RATE))
    
    return jsonify({'image': img_base64, 'id': id_param, 'direction': direction,'images': images_data,'time_video': time_video})


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
