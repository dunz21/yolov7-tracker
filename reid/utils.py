import os
import cv2
import csv
import numpy as np
import utils.PersonImage as PersonImage

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

def resize_with_padding(img, target_size):
    """
    Resize an image with padding to maintain aspect ratio.

    Parameters:
        img (ndarray): The input image.
        target_size (tuple): The target size as (width, height).
    
    Returns:
        ndarray: The resized image with added padding.
    """
    target_width, target_height = target_size
    height, width = img.shape[:2]

    # Calculate scale to resize the image
    scale = min(target_width / width, target_height / height)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    resized_img = cv2.resize(img, (resized_width, resized_height))

    # Create a white background
    background = np.full((target_height, target_width, 3), 255, dtype=np.uint8)

    # Calculate padding sizes
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2

    # Place the resized image on the white background
    background[y_offset:y_offset+resized_height, x_offset:x_offset+resized_width] = resized_img

    return background

def save_image_grid(sub_frames,name='images_subframe',id='', resize=None):
    """
    Save a grid of images with 6 images per row.
    
    Parameters:
        sub_frames (list of ndarray): List of images (sub_frames) to be concatenated into a grid.
        grid_name (str): Name of the output grid image.
        resize (tuple): Optional. New size as (width, height) to resize images.
    """
    rows = []
    for i in range(0, len(sub_frames), 6):  # Process 6 images at a time
        row = sub_frames[i:i+6]
        # Optionally resize images
        if resize is not None:
            row = [cv2.resize(img, resize) for img in row]
        # Make sure the row has 6 images, add black images if necessary
        while len(row) < 6:
            if resize is not None:
                empty_img = np.zeros((resize[1], resize[0], 3), dtype=np.uint8)
            else:
                height, width = row[0].shape[:2]
                empty_img = np.zeros((height, width, 3), dtype=np.uint8)
            row.append(empty_img)
        # Concatenate images in the row horizontally
        concatenated_row = np.hstack(row)
        rows.append(concatenated_row)
    # Concatenate all rows vertically to create the grid
    grid_image = np.vstack(rows)
    


    id_directory = os.path.join(f"{name}", str(id))
    if not os.path.exists(id_directory):
        os.makedirs(id_directory)
    image_name = f"img_{id}_{len(sub_frames)}_total.png"
    save_path = os.path.join(id_directory, image_name)
    cv2.imwrite(save_path, grid_image)
    return save_path


def save_csv_bbox(personImage:PersonImage, filename='bbox.csv'):
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

def save_csv_bbox_alternative(personImage:PersonImage, filename='bbox.csv'):
    # Check if the file exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode ('a') if it exists, otherwise in write mode ('w')
    with open(filename, 'a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file is being created for the first time
        if not file_exists:
            writer.writerow(['id', 'x1', 'y1', 'x2', 'y2', 'centroid_bottom_x', 'centroid_bottom_y','frame_number','overlap','distance_to_center','score'])

        # Append data
        #personImage.list_images.sort(key=lambda x: x.frame_number)
        for img in sorted(personImage.list_images, key=lambda x: x.frame_number):
            x1, y1, x2, y2, score = img.bbox
            centroid_bottom_x = (x1 + x2) // 2
            centroid_bottom_y = y2
            writer.writerow([personImage.id, int(x1), int(y1), int(x2), int(y2), int(centroid_bottom_x), int(centroid_bottom_y),img.frame_number,img.overlap,img.distance_to_center,score])
            