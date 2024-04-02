import os
import cv2
import csv
import numpy as np
import utils.PersonImage as PersonImage
from shapely.geometry import LineString, Polygon, box

def path_intersects_line(centroids, line):
    path = LineString(centroids)
    line = LineString(line)
    return path.intersects(line)

def point_side_of_line(point, line_start, line_end):
    line_vec = [line_end[0] - line_start[0], line_end[1] - line_start[1]]
    point_vec = [point[0] - line_start[0], point[1] - line_start[1]]
    cross_product = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    if cross_product > 0:
        return "In"
    elif cross_product < 0:
        return "Out"
    else:
        return "on the line"
    
def bbox_inside_any_polygon(polygons_points, bbox_tlbr):
    # Convert bbox from tlbr format to a Shapely box
    tl_x, tl_y, br_x, br_y = bbox_tlbr
    bbox = box(tl_x, tl_y, br_x, br_y)
    
    # Iterate over each polygon in the list
    for polygon_points in polygons_points:
        # Convert the current polygon points into a Shapely Polygon
        polygon = Polygon(polygon_points)
        
        # Check if the bbox is completely within the current polygon
        if polygon.contains(bbox):
            return True  # Return True if the bbox is inside any polygon
    
    return False  # Return False if the bbox is not inside any polygon
    
def guess_final_direction(arr, initial_value):
    """
    Removes all occurrences of initial_value from the beginning of the array until a different value is encountered.
    Then calculates the percentage of remaining elements in the array that are equal to initial_value.
    
    Parameters:
    - arr: List of strings, each being "In" or "Out"
    - initial_value: The initial value to remove and then calculate the percentage for ("In" or "Out")
    
    Returns:
    - Percentage of elements equal to initial_value in the remaining array
    """
    # Find the first index where the value is not initial_value
    first_different_index = None
    for i, value in enumerate(arr):
        if value != initial_value:
            first_different_index = i
            break
    
    # If all values are the same as initial_value, the remaining list is empty
    if first_different_index is None:
        return initial_value
    
    # Slice the array to remove the initial_values
    filtered_arr = arr[first_different_index:]
    
    # Count the occurrences of initial_value in the remaining array
    count_initial = filtered_arr.count(initial_value)
    
    # How many values of the initial value I have in the next part of the array
    # If its lower than 20% then I consider that the person is not going in the same direction
    if len(filtered_arr) > 0:
        percentage = (count_initial / len(filtered_arr)) * 100
        if percentage < 20:
            if initial_value == "In":
                return "Out"
            else:
                return "In"
    return initial_value
    



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

def save_csv_bbox_alternative(personImage: PersonImage, filepath='',folder_name='', direction=''):
    EVERY_WHEN_TO_SAVE = 3
    # Check if the folder exists, create it if not
    # if not os.path.exists(BASE_FOLDER_NAME):
        # os.makedirs(BASE_FOLDER_NAME)
    
    # Update the filepath to include the folder path
    # filepath = os.path.join(BASE_FOLDER_NAME, filepath)
    # Check if the file exists
    file_exists = os.path.isfile(filepath)

    # Open the file in append mode ('a') if it exists, otherwise in write mode ('w')
    with open(filepath, 'a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file is being created for the first time
        if not file_exists:
            writer.writerow(['id', 'x1', 'y1', 'x2', 'y2', 'centroid_x', 'centroid_y', 'area', 'frame_number', 'overlap', 'distance_to_center', 'conf_score','img_name'])

        # Append data
        for index, img in enumerate(sorted(personImage.list_images, key=lambda x: x.frame_number)):
            image_name = ''
            if index % EVERY_WHEN_TO_SAVE == 0:
                image_name = save_image_based_on_sub_frame(img.frame_number, img.img_frame, personImage.id, folder_name=folder_name, direction=direction, bbox=img.bbox)
            x1, y1, x2, y2, conf_score = img.bbox
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            overlap_rounded = round(img.overlap, 2)
            distance_to_center_rounded = round(img.distance_to_center, 2)
            conf_score_rounded = round(conf_score, 2)
            writer.writerow([personImage.id, int(x1), int(y1), int(x2), int(y2), int(centroid_x), int(centroid_y), area, img.frame_number, overlap_rounded, distance_to_center_rounded, conf_score_rounded, image_name])

            