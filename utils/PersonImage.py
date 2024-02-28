import numpy as np
import cv2
import sys
from reid.utils import save_image_based_on_sub_frame,save_csv_bbox_alternative
from reid.BoundingBox import BoundingBox
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
    def save(cls, id, folder_name='images_subframe', csv_box_name='bbox.csv'):
        """
            Save the instance with the specified id to a file.
        """
        instance = cls.get_instance(id)
        if instance is None or instance.direction is None or len(instance.history_deque) == 0 or len(instance.list_images) == 0:
            return
        best_image = instance.get_best_images(1)[0]

        for i,img in enumerate(instance.list_images):
            if i % 3 == 0:
                save_image_based_on_sub_frame(img.frame_number, img.img_frame, instance.id, folder_name=folder_name, direction=instance.direction, bbox=img.bbox)


        # save_csv_bbox(personImage=instance, filename=csv_box_name) # Comprobar si esto es realmente necesario
        save_csv_bbox_alternative(personImage=instance, filename=f"{csv_box_name}.csv")
        cls.delete_instance(id)


    def get_best_images(self, n=1):
        """
        Get the best n images from the list based on overlap and distance_to_center. 
        PENDING.....
        """
        # Sort the list based on overlap and distance_to_center
        self.list_images.sort(key=lambda x: (x.overlap, x.distance_to_center))
        # Return the best n images
        return self.list_images[:n]

    
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
    
    def find_polygons_for_centroids(cls,polygons_list):
        """
        This function takes a list of polygons and a list of centroids and returns a list of indices
        to later use it to detect if the person was first inside a polygon and then outside of it.
        """
        if len(cls.history_deque) < 2:
            return None
        if cls.polygons.__len__() == 0:
            cls.polygons = polygons_list
            
        # centroids = [cls.calculate_centroid_bottom_tlbr(bbox) for bbox in cls.history_deque] EX
        centroids = [cls.calculate_centroid(bbox) for bbox in cls.history_deque]
        cls.polygon_indices = []
        
        for centroid_index , centroid in enumerate(centroids):
            for i, polygon in enumerate(cls.polygons):
                if cls.is_point_in_polygon(centroid, polygon): #Entra el primer punto al poligono
                    cls.polygon_indices.append(i)

        return cls.polygon_indices


        