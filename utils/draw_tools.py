import cv2
import os
import numpy as np
from PIL import Image

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
    #Inside 
    RED = (0, 0, 255) #BGR
    RED = (84,27,227) #BGR
    pts_entrance = polygon_in.reshape((-1, 1, 2))
    if image is not None:
        cv2.polylines(image, [pts_entrance], isClosed=True, color=RED, thickness=2)

    #Outside
    BLUE = (255,0,0) #BGR
    BLUE = (116,186,79) #BGR
    pts_exit = polygon_out.reshape((-1, 1, 2))
    if image is not None:
        cv2.polylines(image, [pts_exit], isClosed=True, color=BLUE, thickness=2)
    return [pts_entrance,pts_exit]

def filter_detections_inside_polygon(detections,polygon_pts=np.array([[0,1080],[0,600],[510,500],[593,523],[603,635],[632,653],[738,588],[756,860],[587,1080]], np.int32)):
    """
    Filters detections based on whether the midpoint of the bottom edge of their bounding box
    is inside a specified polygon.

    :param detections: A numpy array of detections with shape (N, 6), where the first four columns
                       represent the bounding box coordinates (x1, y1, x2, y2).
    :param polygon_pts: A numpy array of points defining the polygon, shape (M, 2).
    :return: A numpy array of filtered detections.
    """

    # Convert polygon points to the required shape for cv2.polygonTest
    polygon = polygon_pts.reshape((-1, 1, 2))

    # Function to check if a point is inside the polygon
    def is_point_inside_polygon(point, poly):
        return cv2.pointPolygonTest(poly, point, False) >= 0

    # Filter detections
    filtered_detections = []
    for det in detections:
        # Calculate the midpoint at the bottom of the bounding box
        midpoint_x = (det[0] + det[2]) / 2
        midpoint_y = det[3]
        midpoint = (midpoint_x, midpoint_y)

        # Check if the midpoint is inside the polygon
        if is_point_inside_polygon(midpoint, polygon):
            filtered_detections.append(det)

    return np.array(filtered_detections)

def draw_polygon_interested_area(frame, polygon_pts=np.array([[0,1080],[0,600],[510,500],[593,523],[603,635],[632,653],[738,588],[756,860],[587,1080]], np.int32)):
    polygon_pts = polygon_pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon_pts], isClosed=True, color=(0, 255, 0), thickness=1)

