import numpy as np
import cv2

class PersonImage:
    _instances = {}  # Class-level dictionary to store instances

    def __new__(cls, id, list_images, direction, history_deque =[],polygons=[]):
        # Check if an instance with the given id already exists
        if id in cls._instances:
            cls._instances[id].list_images.extend(list_images)
            cls._instances[id].direction = direction
            cls._instances[id].history_deque = history_deque
            return cls._instances[id]
        else:
            # Create a new instance and store it in the dictionary
            instance = super(PersonImage, cls).__new__(cls)
            cls._instances[id] = instance
            return instance

    def __init__(self, id, list_images, direction, history_deque=[],polygons=[]):
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

    # Optionally, a method to retrieve an instance by ID
    @classmethod
    def get_instance(cls, id):
        return cls._instances.get(id)
    
    @classmethod
    def calculate_centroid(cls,tlwh):
        x, y, w, h = tlwh
        return np.array([x + w / 2, y + h])
    
    @classmethod
    def calculate_centroid_tlbr(cls,tlbr):
        x1, x2, y1, y2 = tlbr
        midpoint_x = (x1 + y1) / 2
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


    def find_polygons_for_centroids(cls,polygons_list):
        if len(cls.history_deque) < 2:
            return None
        if cls.polygons.__len__() == 0:
            cls.polygons = polygons_list
            cls.is_in_polygon = [False for _ in range(len(cls.polygons))]
            
        # centroids = [cls.calculate_centroid(bbox) for bbox in reversed(cls.history_deque)]
        centroids = [cls.calculate_centroid_tlbr(bbox) for bbox in reversed(cls.history_deque)]
        cls.polygon_indices = []
        
        exit = False
        for centroid_index , centroid in enumerate(centroids):
            in_any_polygon = False
            for i, polygon in enumerate(cls.polygons):
                if cls.is_point_in_polygon(centroid, polygon): #Entra el primer punto al poligono
                    cls.is_in_polygon[i] = True # Alguna vez ya pase por este poligono
                    cls.polygon_indices.append(i)
                    in_any_polygon = True # En esta iteracion por lo menos pase

            
            if all(value == False for value in cls.is_in_polygon) and centroid_index == 0:
                # Esto quiere decir que el primer centroid no ha tocado ningun poligono, por ende los de atras tampoco
                return None 
            if all(cls.is_in_polygon) and in_any_polygon == False and centroid_index == 0:
                exit = True
            

        return {
            'exit': exit,
            'polygon_indices': cls.polygon_indices,
            'direction': cls.calculate_direction(cls.polygon_indices),
            'between_polygons': cls.between_polygons(cls.polygon_indices), # La idea es para sacar fotos justo cuando este en una transicion. Revisar que funcione
        }


    # def find_polygons_for_centroids(cls,polygons_list):
    #     if len(cls.history_deque) < 2:
    #         return None
    #     if cls.polygons.__len__() == 0:
    #         cls.polygons = polygons_list
    #         cls.is_in_polygon = [False for _ in range(len(cls.polygons))]
            
    #     centroids = [cls.calculate_centroid(bbox) for bbox in reversed(cls.history_deque)]
    #     cls.polygon_indices = []
        
    #     for centroid_index , centroid in enumerate(centroids):
    #         in_any_polygon = False
    #         exit = False
    #         for i, polygon in enumerate(cls.polygons):
    #             if cls.is_point_in_polygon(centroid, polygon): #Entra el primer punto al poligono
    #                 cls.polygon_indices.append(i)
    #                 cls.is_in_polygon[i] = True
    #                 in_any_polygon = True
    #                 print(cls.is_in_polygon)
    #             elif all(cls.is_in_polygon): #Si pasaron por todos los poligonos y este punto ya no esta 
    #                 pass
    #                 # exit = True
    #                 # return {
    #                 #     'exit': True,
    #                 #     'polygon_indices': cls.polygon_indices,
    #                 #     'direction': cls.calculate_direction(cls.polygon_indices),
    #                 #     'between_polygons': cls.between_polygons(cls.polygon_indices), # La idea es para sacar fotos justo cuando este en una transicion. Revisar que funcione
    #                 # }
    #             elif any(cls.is_in_polygon): #Si ya pasaron por algun poligono y este punto no esta en ninguno
    #                 continue
    #             elif all(value == False for value in cls.is_in_polygon):
    #                 # Significa que el primer centroid no ha tocado ningun poligono
    #                 return None
            
    #         if all(cls.is_in_polygon) and in_any_polygon == False and centroid_index == 0:
    #             exit = True

    #     return {
    #         'exit': exit,
    #         'polygon_indices': cls.polygon_indices,
    #         'direction': cls.calculate_direction(cls.polygon_indices),
    #         'between_polygons': cls.between_polygons(cls.polygon_indices), # La idea es para sacar fotos justo cuando este en una transicion. Revisar que funcione
    #     }
                    

        # Esta funcion yo creo que la voy a sacar por que la logica es mirar la mitad
        # Y casi nunca es el caso....
        # Tiene que mirar la transicion y por ej de 0 a 1 y despues mirar otros frames mas y listo
        # if cls.between_polygons(cls.polygon_indices):
        # #     cls.has_been_between_polygons = True

        # if not found_in_polygon:
        #     return None
        
        # direction = cls.calculate_direction(cls.polygon_indices)
        # if direction is not None:
        #     cls.direction = direction

        