class BoundingBox:
    def __init__(self, img_frame, frame_number,bbox,overlap,distance_to_center):
        self.img_frame = img_frame
        self.frame_number = frame_number
        self.bbox = bbox
        self.overlap = overlap
        self.distance_to_center = distance_to_center


    