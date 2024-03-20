import numpy as np
import cv2
import sys
from reid.utils import save_csv_bbox_alternative,path_intersects_line,point_side_of_line,guess_final_direction
from reid.BoundingBox import BoundingBox
from shapely.geometry import Point, Polygon, LineString
import os
class PersonImage:
    _instances = {}  # Class-level dictionary to store instances
    _max_instances = 1000  # Max number of instances to store

    def __new__(cls, id=0, list_images=[], history_deque =[],polygons=[]):

        if len(cls._instances) >= cls._max_instances:
            # Remove the oldest instance
            oldest_id = sorted(cls._instances.keys())[0]
            del cls._instances[oldest_id]
            
        # Check if an instance with the given id already exists
        if id in cls._instances:
            cls._instances[id].list_images.extend(list_images)
            cls._instances[id].history_deque = history_deque
            return cls._instances[id]
        else:
            # Create a new instance and store it in the dictionary
            instance = super(PersonImage, cls).__new__(cls)
            cls._instances[id] = instance
            return instance

    def __init__(self, id, list_images=[], direction=None, history_deque=[],polygons=[]):
        # Initialize only if the instance is new
        if not hasattr(self, '_initialized'):
            self.id = id
            self.list_images = list_images
            self.direction = direction
            self.polygons = polygons
            self.history_deque = history_deque
            self.polygon_indices = []
            self.list_features = []
            self.ready = False
            self._initialized = True  # Mark as initialized


    @classmethod
    def save(cls, id, folder_name='images_subframe', csv_box_name='bbox.csv', polygons_list=[]):
        """
            Save the instance with the specified id to a file.
        """
        instance = cls.get_instance(id)
        
        if id == 3:
            print(f"Cross green line")
    
        if instance is None or len(instance.history_deque) < 5 or len(instance.list_images) < 5:
            return
        
        
        #### Si soy FALSO IN, y comparo con CENTER CENTROID y es OUT, entonces es OUT
        #### Si soy FALSO OUT, y comparo con CENTER CENTROID y es IN, entonces es IN
        
        centroids = [(cls.calculate_centroid_bottom_tlbr(bbox), cls.calculate_centroid(bbox)) for bbox in instance.history_deque]
        centroid_bottom, centroid_center = zip(*centroids) if centroids else ([], [])
        cross_green_line = path_intersects_line(centroid_bottom, LineString(polygons_list[0][:2])) or path_intersects_line(centroid_bottom, LineString(polygons_list[0][2:]))
        
        if id == 3:
            print(f"Cross green line: {cross_green_line}")
            
        if instance is None or cross_green_line is False:
            return
        
        initial_direction_bottom = point_side_of_line(np.mean(centroid_bottom[:2],axis=0), polygons_list[0][0], polygons_list[0][1])
        final_direction_bottom = point_side_of_line(np.mean(centroid_bottom[-2:], axis=0), polygons_list[0][0], polygons_list[0][1])
        
        initial_direction = initial_direction_bottom
        final_direction = final_direction_bottom
        
        
        initial_direction_center = point_side_of_line(np.mean(centroid_bottom[:2],axis=0), polygons_list[0][0], polygons_list[0][1])
        # final_direction_center = point_side_of_line(np.mean(centroid_bottom[-2:], axis=0), polygons_list[0][0], polygons_list[0][1])
        
        # Esto es cuando falso IN con el bottom centroid pero el del center me dice que es OUT
        if (initial_direction_bottom == 'In') & (initial_direction_center == 'Out'):
            initial_direction = 'Out'
        
    
        
        
        if final_direction == initial_direction:
            return
            # total_in_out = [point_side_of_line(centroid, polygons_list[0][0], polygons_list[0][1]) for centroid in centroid_bottom]
            # new_guess_final_direction = guess_final_direction(total_in_out, initial_direction)
            # if new_guess_final_direction == final_direction:        
            #     return
            # direction = new_guess_final_direction
            # os.makedirs(folder_name, exist_ok=True)
            # with open(f'{folder_name}/{csv_box_name}.txt', 'a') as log_file:
            #     log_file.write(f"ID: {id} - Total: {total_in_out} - New guess: {new_guess_final_direction}\n")
        
        if final_direction == "Out" and initial_direction == "In":
            direction = "Out"
        elif final_direction == "In" and initial_direction == "Out":
            direction = "In"
        else:
            direction = "None"

        save_csv_bbox_alternative(personImage=instance, filename=f"{csv_box_name}.csv",folder_name=folder_name, direction=direction)
        cls.delete_instance(id)
    
    @classmethod
    def clear_instances(cls):
        """
        Clears all stored instances from the _instances dictionary.
        """
        cls._instances.clear()

    @classmethod
    def delete_instance(cls, id):
        """
        Deletes an instance with the specified id from the _instances dictionary.
        """
        if id in cls._instances:
            del cls._instances[id]

    @classmethod
    def get_memory_usage(cls):
        total_size = 0
        for instance in cls._instances.values():
            total_size += sys.getsizeof(instance)
        return total_size

    # Optionally, a method to retrieve an instance by ID
    @classmethod
    def get_instance(cls, id):
        return cls._instances.get(id)
    
    @classmethod
    def calculate_centroid(cls,tlbr):
        x1, y1, x2, y2 = tlbr
        midpoint_x = (x1 + x2) // 2
        midpoint_y = (y1 + y2) // 2
        midpoint = (midpoint_x, midpoint_y)
        return midpoint

    
    @classmethod
    def calculate_centroid_bottom_tlbr(cls,tlbr):
        x1, y1, x2, y2 = tlbr
        midpoint_x = (x1 + x2) // 2
        midpoint_y = y2
        midpoint = (midpoint_x, midpoint_y)
        return midpoint

    @classmethod
    def is_point_in_polygon(cls,point, polygon):
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

    @classmethod
    def between_polygons(cls,arr):
        if(arr.__len__() < 2):
            return None
        # Calculate middle index
        middle_index = len(arr) // 2
        
        # Adjust for even-length arrays, if needed
        if len(arr) % 2 == 0:
            middle_index -= 1
        
        # Determine start and end indices for the subarray
        start = max(0, middle_index - 4)
        end = min(len(arr), middle_index + 5)
        
        # Extract the subarray
        subarray = arr[start:end]
        
        # Count the number of 1s and 0s
        count_ones = subarray.count(1)
        count_zeros = subarray.count(0)
        
        # Check if they are equal
        return count_ones == count_zeros
    
    @classmethod
    def is_line_in_polygon(cls, line, polygon):
        return line.intersects(polygon)
    
    @classmethod
    def calculate_direction(cls,polygon_indices):
        """
        This function takes an array of 1s and 0s and returns a string representing
        the point where the transition from 1 to 0 or 0 to 1 happens.
        """
        if(polygon_indices.__len__() < 2):
            return None
        for i in range(len(polygon_indices) - 1):
            if polygon_indices[i] != polygon_indices[i + 1]:
                if polygon_indices[i] == 1:
                    return 'Out'
                else:
                    return 'In'
        return None
    
    def detect_pattern_change(cls,index_list):
        # Initialize variables to track the last value and the position of change
        last_value = None
        change_position = -1
        if index_list is None:
            return None

        for i, index in enumerate(index_list):
            if last_value is not None and index != last_value:
                # Detect the change
                change_type = f"{last_value}{index}"
                change_position = i - 1
                break
            last_value = index

        # If no change was detected
        if change_position == -1:
            return None

        # Calculate the positions forward from the change
        positions_forward = len(index_list) - change_position - 1
        return change_type, positions_forward

    def is_bbox_in_polygon(cls,bbox,polygons_list):
        centroid = cls.calculate_centroid_bottom_tlbr(bbox)
        inside_any_polygon = False
        for polygon in polygons_list:
            if cls.is_point_in_polygon(centroid, polygon):
                inside_any_polygon = True
        return inside_any_polygon
    
    def find_polygons_for_centroids(cls, polygons_list):
        if len(cls.history_deque) < 2:
            return None
        if not cls.polygons:
            shapely_polygons = [Polygon(polygon.reshape(-1, 2)) for polygon in polygons_list]
            cls.polygons = [Polygon(polygon) for polygon in shapely_polygons]
        
        centroids = [cls.calculate_centroid(bbox) for bbox in cls.history_deque]
        bottom_centroids = [cls.calculate_centroid_bottom_tlbr(bbox) for bbox in cls.history_deque]
        
        cls.polygon_indices = []
        
        for centroid_index, centroid in enumerate(centroids):
            line = LineString([centroid, bottom_centroids[centroid_index]])
            for i, polygon in enumerate(cls.polygons):
                if cls.is_line_in_polygon(line, polygon):  # Checks if the line enters the polygon
                    cls.polygon_indices.append(i)

        return cls.polygon_indices


        