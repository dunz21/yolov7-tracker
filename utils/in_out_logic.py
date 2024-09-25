import cv2
import numpy as np
from utils.types import Direction

def calculate_bbox_center(bbox):
    """
    Calculates the center point of a bounding box.
    bbox: [x1, y1, x2, y2]
    Returns: (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return (center_x, center_y)

def draw_movement_vector(frame, bbox_history, id=None, color=(0, 255, 0)):
    """
    Draws the movement vector on the frame.
    """
    if len(bbox_history) < 2:
        return frame
    centers = [calculate_bbox_center(bbox) for bbox in bbox_history]
    # Draw lines between consecutive centers
    for i in range(len(centers)-1):
        pt1 = (int(centers[i][0]), int(centers[i][1]))
        pt2 = (int(centers[i+1][0]), int(centers[i+1][1]))
        cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.3)
    # Optionally, you can put the ID near the last point
    if id is not None:
        last_pt = (int(centers[-1][0]), int(centers[-1][1]))
        cv2.putText(frame, f'ID:{id}', last_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def calculate_average_movement_vector(bbox_history):
    """
    Calculates the average movement vector from a sequence of bounding boxes.
    """
    if len(bbox_history) < 2:
        return None
    centers = [calculate_bbox_center(bbox) for bbox in bbox_history]
    # Calculate displacement vectors between consecutive centers
    displacement_vectors = [np.array(centers[i+1]) - np.array(centers[i]) for i in range(len(centers)-1)]
    # Compute the average movement vector
    avg_movement_vector = np.mean(displacement_vectors, axis=0)
    return avg_movement_vector

def calculate_average_movement_vector(bbox_history):
    """
    Calculates the average movement vector from a sequence of bounding boxes.
    """
    if len(bbox_history) < 2:
        return None
    centers = [calculate_bbox_center(bbox) for bbox in bbox_history]
    # Calculate displacement vectors between consecutive centers
    displacement_vectors = [np.array(centers[i+1]) - np.array(centers[i]) for i in range(len(centers)-1)]
    # Compute the average movement vector
    avg_movement_vector = np.mean(displacement_vectors, axis=0)
    return avg_movement_vector

def determine_direction_from_vector(movement_vector, entrance_line):
    """
    Determines the direction (In or Out) based on the movement vector and entrance line.
    movement_vector: [dx, dy]
    entrance_line: [(x1, y1), (x2, y2)] - two points defining the entrance line
    Returns: Direction.In, Direction.Out, or Direction.Undefined
    """
    if movement_vector is None or np.linalg.norm(movement_vector) == 0:
        return Direction.Undefined.value
    # Calculate the normal vector to the entrance line
    x1, y1 = entrance_line[0]
    x2, y2 = entrance_line[1]
    line_vector = np.array([x2 - x1, y2 - y1])
    normal_vector = np.array([-line_vector[1], line_vector[0]])  # Rotate by 90 degrees
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize

    # Normalize movement vector
    movement_vector_norm = movement_vector / np.linalg.norm(movement_vector)

    # Calculate the dot product
    dot_product = np.dot(movement_vector_norm, normal_vector)

    # Determine direction based on the sign of the dot product
    threshold = 0.3  # You can adjust this threshold
    if dot_product > threshold:
        return Direction.In.value
    elif dot_product < -threshold:
        return Direction.Out.value
    else:
        return Direction.Undefined.value