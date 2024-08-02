class InferenceParams:
    def __init__(self, weights_folder='yolov7.pt', yolo_model_version='yolov7', tracker='sort', save_all_images=False):
        self.weights_folder = weights_folder
        self.yolo_model_version = yolo_model_version
        self.tracker = tracker
        self.save_all_images = save_all_images
        