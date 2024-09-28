import numpy as np
import cv2
import sys
from reid.utils import save_csv_bbox_alternative,path_intersects_line,point_side_of_line,guess_final_direction,bbox_inside_any_polygon
from reid.BoundingBox import BoundingBox
from shapely.geometry import Point, Polygon, LineString
from IPython import embed
from utils.types import Direction
from utils.in_out_logic import calculate_average_movement_vector,determine_direction_from_vector
import random

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

    def addBoundingBox(self, objBbox, max_images=20):
        """
        Adds a BoundingBox to the list and ensures that the total number of
        BoundingBoxes with non-None img_frame does not exceed max_images.
        
        Parameters:
        - objBbox: The BoundingBox object to add.
        - max_images: The maximum allowed number of BoundingBoxes with an img_frame.
        """
        # Append the new BoundingBox to the list
        self.list_images.append(objBbox)

        # Count the number of BoundingBoxes with non-None img_frame
        num_images = sum(1 for bbox in self.list_images if bbox.img_frame is not None)

        # If the count exceeds max_images, remove img_frame from a random BoundingBox
        if num_images > max_images:
            # Get indices of BoundingBoxes with non-None img_frame
            indices = [i for i, bbox in enumerate(self.list_images) if bbox.img_frame is not None]
            # Randomly select one index to remove img_frame
            idx_to_remove = random.choice(indices)
            # Set img_frame to None to free up memory
            self.list_images[idx_to_remove].img_frame = None
            
    @classmethod
    def save(cls, id, folder_name='images_subframe', csv_box_name='bbox.csv', polygons_list=[],FPS=15, save_img=True, save_all=False,bbox_centroid=None):
        """
            Save the instance with the specified id to a file.
        """
        instance = cls.get_instance(id)
    
        if instance is None or len(instance.history_deque) < 2 or len(instance.list_images) < 2:
            return
        #
        
        
        #Calculate the average movement vector
        bbox_history = list(instance.history_deque)
        avg_movement_vector = calculate_average_movement_vector(bbox_history)

        # Define the entrance line (assumed to be polygons_list[0])
        entrance_line = polygons_list[0][:2]  # [(x1, y1), (x2, y2)]

        # Determine direction
        new_direction = determine_direction_from_vector(avg_movement_vector, entrance_line)
        crossed_zone = any(bbox_inside_any_polygon(polygons_list, bbox) for bbox in bbox_history)
        #### Si soy FALSO IN, y comparo con CENTER CENTROID y es OUT, entonces es OUT
        #### Si soy FALSO OUT, y comparo con CENTER CENTROID y es IN, entonces es IN
        
        centroids = [(cls.calculate_centroid_bottom_tlbr(bbox), cls.calculate_centroid_top(bbox)) for bbox in instance.history_deque]
        centroid_bottom, centroid_top = zip(*centroids) if centroids else ([], [])
        
        if bbox_centroid is not None and bbox_centroid.lower() == 'top':
            centroid_bottom = centroid_top
        
        cross_green_line = path_intersects_line(centroid_bottom, LineString(polygons_list[0][:2]))
        
        if instance is None or cross_green_line is False:
            # No cruzo la linea verde, pero toque alguno de los polygonos
            # Se supone que la unica forma de entrar aca, y seria solo 1 vez (se supone) es que aparezanas primero en remove tracks
            for bbox in instance.history_deque:
                if bbox_inside_any_polygon(polygons_list, bbox):
                    save_csv_bbox_alternative(personImage=instance, filepath=csv_box_name,folder_name=folder_name, direction=Direction.Undefined.value,FPS=FPS,save_img=False,new_direction=new_direction)
                    cls.delete_instance(id)
                    return
            return
        
        initial_direction_bottom = point_side_of_line(np.mean(centroid_bottom[:2],axis=0), polygons_list[0][0], polygons_list[0][1])
        final_direction_bottom = point_side_of_line(np.mean(centroid_bottom[-2:], axis=0), polygons_list[0][0], polygons_list[0][1])
        
        initial_direction = initial_direction_bottom
        final_direction = final_direction_bottom
        
        
        initial_direction_center = point_side_of_line(np.mean(centroid_bottom[:2],axis=0), polygons_list[0][0], polygons_list[0][1])
        # final_direction_center = point_side_of_line(np.mean(centroid_bottom[-2:], axis=0), polygons_list[0][0], polygons_list[0][1])
        
        # Esto es cuando falso IN con el bottom centroid pero el del center me dice que es OUT
        if (initial_direction_bottom == Direction.In.value) & (initial_direction_center == Direction.Out.value):
            initial_direction = Direction.Out.value

        
        if final_direction == initial_direction:
            # Cruzo la linea verde pero tiene direccion indecisa.
            # Guardar igualmente como direccion indecisa
            save_csv_bbox_alternative(personImage=instance, filepath=csv_box_name,folder_name=folder_name, direction=Direction.Cross.value,FPS=FPS, save_img=save_img,new_direction=new_direction)
            cls.delete_instance(id)
            return
            # total_in_out = [point_side_of_line(centroid, polygons_list[0][0], polygons_list[0][1]) for centroid in centroid_bottom]
            # new_guess_final_direction = guess_final_direction(total_in_out, initial_direction)
            # if new_guess_final_direction == final_direction:        
            #     return
            # direction = new_guess_final_direction
            # os.makedirs(folder_name, exist_ok=True)
            # with open(f'{folder_name}/{csv_box_name}.txt', 'a') as log_file:
            #     log_file.write(f"ID: {id} - Total: {total_in_out} - New guess: {new_guess_final_direction}\n")
        
        if final_direction == Direction.Out.value and initial_direction == Direction.In.value:
            direction = Direction.Out.value
        elif final_direction == Direction.In.value and initial_direction == Direction.Out.value:
            direction = Direction.In.value
        else:
            direction = Direction.Undefined.value
        save_csv_bbox_alternative(personImage=instance, filepath=csv_box_name,folder_name=folder_name, direction=direction,FPS=FPS, save_img=save_img,new_direction=new_direction)
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
    def calculate_centroid_top(cls, tlbr):
        x1, y1, x2, y2 = tlbr
        midpoint_x = (x1 + x2) // 2
        midpoint_y = y1
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
                    return Direction.Out.value
                else:
                    return Direction.In.value
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
        