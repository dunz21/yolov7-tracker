class VideoOption:
    def __init__(self, folder_results='',view_img=False, noSaveVideo=False, save_img_bbox=True, weights='yolov7.pt', model_version='yolov10'):
        self.weights = weights
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = '0'
        self.save_txt = False
        self.classes = None
        self.agnostic_nms = False
        self.model_version = model_version
        self.augment = False
        self.update = False
        self.project = folder_results
        self.exist_ok = False
        self.no_trace = False
        self.save_bbox_dim = False
        self.save_with_object_id = False
        self.download = True
        self.save_img_bbox = save_img_bbox # Save every bouding box
        self.show_config = True # Show tracker config
        self.nosave = noSaveVideo # GUARDAR VIDEO, True para NO GUARDAR
        self.view_img = view_img # DEBUG IMAGE
        self.wait_for_key = False # DEBUG KEY