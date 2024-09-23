import cv2
import os
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, Polygon, box
from reid.utils import point_side_of_line

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

def draw_boxes_entrance_exit(image=None,polygon_in=np.array([[265, 866],[583, 637],[671, 686],[344, 948]], np.int32),polygon_out=np.array([[202, 794],[508, 608],[575, 646],[263, 865]], np.int32)):
    # 0 -> 1 means exit
    # 1 -> 0 means entrance
    #Inside Green
    GREEN = (116,186,79) #BGR
    pts_entrance = polygon_in.reshape((-1, 1, 2))
    if image is not None:
        cv2.polylines(image, [pts_entrance], isClosed=True, color=GREEN, thickness=2)

    #Outside Red
    RED = (84,27,227) #BGR
    pts_exit = polygon_out.reshape((-1, 1, 2))
    if image is not None:
        cv2.polylines(image, [pts_exit], isClosed=True, color=RED, thickness=2)
    return [pts_entrance,pts_exit]


def filter_detections_inside_polygon(detections, polygon_pts=np.array([[0,1080],[0,600],[510,500],[593,523],[603,635],[632,653],[738,588],[756,860],[587,1080]], np.int32), bbox_complete_match=False):
    """
    Filters detections based on whether the midpoint of the bottom edge of their bounding box
    or any part of the bounding box if bbox_complete_match=True, is touching or inside a specified polygon.

    :param detections: A numpy array of detections with shape (N, 6), where the first four columns
                       represent the bounding box coordinates (x1, y1, x2, y2).
    :param polygon_pts: A numpy array of points defining the polygon, shape (M, 2).
    :param bbox_complete_match: A boolean that if True, checks if any part of the bbox touches the polygon.
    :return: A numpy array of filtered detections.
    """

    # Convert polygon points to the required shape for cv2.polygonTest
    polygon = polygon_pts.reshape((-1, 1, 2))

    # Function to check if a point is inside the polygon
    def is_point_inside_polygon(point, poly):
        return cv2.pointPolygonTest(poly, point, False) >= 0

    # Function to check if a bounding box intersects the polygon
    def does_bbox_touch_polygon(bbox, poly_pts):
        # Convert bbox to a polygon (list of four points)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        bbox_polygon = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], np.int32).reshape((-1, 1, 2))
        
        # Use cv2 to check if the convex hull of the bbox intersects the polygon
        intersect_area, _ = cv2.intersectConvexConvex(bbox_polygon, poly_pts)
        
        return intersect_area > 0

    # Filter detections
    filtered_detections = []
    for det in detections:
        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]

        if bbox_complete_match:
            # Check if the bounding box intersects or touches the polygon
            if does_bbox_touch_polygon([x1, y1, x2, y2], polygon):
                filtered_detections.append(det)
        else:
            # Calculate the midpoint at the bottom of the bounding box
            midpoint_x = (x1 + x2) / 2
            midpoint_y = y2
            midpoint = (midpoint_x, midpoint_y)

            # Check if the midpoint is inside the polygon
            if is_point_inside_polygon(midpoint, polygon):
                filtered_detections.append(det)

    return np.array(filtered_detections)


def filter_model_detector_output(yolo_output, min_area=10000, specific_area_coords=[], overlap_threshold=0.5):
    """
    Filters YOLO output bounding boxes based on minimum area and overlap with a specific area.
    
    Parameters:
    yolo_output (ndarray): The YOLO output array.
    min_area (float): The minimum area threshold for bounding boxes.
    specific_area_coords (list): List of (x, y) tuples defining the specific area polygon.
    overlap_threshold (float): The overlap threshold for filtering (default is 0.5).

    Returns:
    ndarray: Filtered YOLO output.
    """
    filtered_output = []
    specific_area = Polygon(specific_area_coords)
    
    for bbox in yolo_output:
        x1, y1, x2, y2, score, class_id = bbox
        bbox_polygon = box(x1, y1, x2, y2)
        bbox_area = bbox_polygon.area

        if bbox_area < min_area:
            continue
        
        intersection_area = bbox_polygon.intersection(specific_area).area
        if intersection_area / bbox_area >= overlap_threshold:
            continue
        
        filtered_output.append(bbox)
    
    return np.array(filtered_output, dtype=np.float32)

def draw_polygon_interested_area(frame, polygon_pts=np.array([[0,1080],[0,600],[510,500],[593,523],[603,635],[632,653],[738,588],[756,860],[587,1080]], np.int32)):
    polygon_pts = polygon_pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon_pts], isClosed=True, color=(0, 255, 0), thickness=1)
    
def calculate_overlap(rect1, rect2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both rectangles
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate union area
    union_area = area1 + area2 - intersection_area

    # Calculate the overlap percentage
    overlap = intersection_area / union_area

    return overlap

def distance_to_bbox_bottom_line(line=[], bbox=[]):
    """
    Calculate the distance between the closest point on a line and the center of the bottom edge of a bounding box (bbox).
    
    :param line: A list of points [[x1, y1], [x2, y2]] defining the line.
    :param bbox: A tuple representing the bounding box (x1, y1, x2, y2).
    :return: The shortest distance between the line and the center of the bottom edge of the bbox.
    """
    line_obj = LineString(line)
    x1, y1, x2, y2 = bbox

    # Calculate the center of the bottom edge of the bbox
    bottom_center = Point((x1 + x2) / 2, y2)

    # Calculate the shortest distance from the bottom center to the line
    distance = bottom_center.distance(line_obj)
    positive_negative = point_side_of_line([(x1 + x2) / 2,y2], line[0], line[1])
    if positive_negative == 'Out':
        distance = -distance
    return distance
  
def draw_configs(frame, configs, scale=1920):
    if scale < 1920:
        # Coordinates and sizes for a 1280x720 resolution
        x1, y1 = 7, 7  # Adjusted for smaller screen
        w = 166  # Adjusted width
        h = len(configs) * 16 + 7  # Adjusted height
        font_scale = 0.4
        line_height = 16
    else:
        # Coordinates and sizes for a 1920x1080 resolution
        x1, y1 = 10, 10  # 10 pixels from the top and left edge for padding
        w = 250  # Default width for the rectangle
        h = len(configs) * 25 + 10  # Calculate height based on number of configs
        font_scale = 0.6
        line_height = 25

    # The color of the rectangle: Black with full opacity
    color = (0, 0, 0, 255)
    # Draw the rectangle that will contain the stats
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, -1)

    # Initial position to start drawing the text
    text_y = y1 + 20 if scale >= 1920 else y1 + 12  # Adjust start position based on scale

    # Iterate through the dictionary and draw each stat
    for key, value in configs.items():
        label = f"{key}: {value}"
        # Draw the text
        cv2.putText(frame, label, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, [255, 255, 255], 1)
        # Move down to the next line
        text_y += line_height

    return frame

def draw_boxes(img, bbox , offset=(0, 0),extra_info=None,color=None,position='Top'):
    for box in bbox:
        x1, y1, x2, y2,id,score = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        # id = int(id)
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # id = int(identities[i]) if identities is not None else 0

        label = str(id) + ":" + "person"
        if extra_info is not None:
            label += str(f"s:{score:.2f}")
            label += str(f"oc:{extra_info[id]['overlap']:.2f}")
            label += str(f"di:{extra_info[id]['distance']:.2f}")

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if color is None:
            color = (255, 0, 20)
            # color_rect_text = (255, 144, 30)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if position == 'Top':
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        else:
            cv2.rectangle(img, (x1, y2 - 20), (x1 + w, y2), color, -1)
            cv2.putText(img, label, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

    return img