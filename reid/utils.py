import os
import cv2
import csv
import numpy as np
import utils.PersonImage as PersonImage

BASE_FOLDER_NAME = 'logs'

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


def save_csv_bbox(personImage:PersonImage, filename='bbox.csv'):
    # Check if the folder exists, create it if not
    if not os.path.exists(BASE_FOLDER_NAME):
        os.makedirs(BASE_FOLDER_NAME)
    
    # Update the filename to include the folder path
    filename = os.path.join(BASE_FOLDER_NAME, filename)

    # Check if the file exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode ('a') if it exists, otherwise in write mode ('w')
    with open(filename, 'a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file is being created for the first time
        if not file_exists:
            writer.writerow(['id', 'x1', 'y1', 'x2', 'y2', 'centroid_bottom_x', 'centroid_bottom_y'])

        # Append data
        for bbox in personImage.history_deque:
            x1, y1, x2, y2 = bbox
            centroid_bottom_x = (x1 + x2) // 2
            centroid_bottom_y = y2
            writer.writerow([personImage.id, int(x1), int(y1), int(x2), int(y2), int(centroid_bottom_x), int(centroid_bottom_y)])

def save_csv_bbox_alternative(personImage: PersonImage, filename=''):
    # Check if the folder exists, create it if not
    if not os.path.exists(BASE_FOLDER_NAME):
        os.makedirs(BASE_FOLDER_NAME)
    
    # Update the filename to include the folder path
    filename = os.path.join(BASE_FOLDER_NAME, filename)
    # Check if the file exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode ('a') if it exists, otherwise in write mode ('w')
    with open(filename, 'a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file is being created for the first time
        if not file_exists:
            writer.writerow(['id', 'x1', 'y1', 'x2', 'y2', 'centroid_x', 'centroid_y', 'area', 'frame_number', 'overlap', 'distance_to_center', 'conf_score'])

        # Append data
        for img in sorted(personImage.list_images, key=lambda x: x.frame_number):
            x1, y1, x2, y2, conf_score = img.bbox
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            overlap_rounded = round(img.overlap, 2)
            distance_to_center_rounded = round(img.distance_to_center, 2)
            conf_score_rounded = round(conf_score, 2)
            writer.writerow([personImage.id, int(x1), int(y1), int(x2), int(y2), int(centroid_x), int(centroid_y), area, img.frame_number, overlap_rounded, distance_to_center_rounded, conf_score_rounded])

            