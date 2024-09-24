class InferenceParams:
    def __init__(self,inference_params_id='',inference_params_name='',weights_folder='yolov7.pt', yolo_model_version='yolov7', tracker='sort', save_all_images=False,bbox_centroid=None):
        self.inference_params_id= inference_params_id
        self.inference_params_name= inference_params_name
        self.weights_folder = weights_folder
        self.yolo_model_version = yolo_model_version
        self.tracker = tracker
        self.save_all_images = save_all_images  #Es util solo en el video de la puerta, donde se requiere guardar todas las imagenes, y no es necesario
        self.bbox_centroid = bbox_centroid # TOP | BOTTOM
        