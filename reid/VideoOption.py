class VideoOption:
    def __init__(self):
        self.weights = 'yolov7.pt'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = '0'
        self.save_txt = False
        self.classes = [0]
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.project = 'runs/detect'
        self.exist_ok = False
        self.no_trace = False
        self.save_bbox_dim = False
        self.save_with_object_id = False
        self.download = True
        self.show_config = True # Show tracker config
        self.nosave = False # GUARDAR VIDEO, True para NO GUARDAR
        self.view_img = False # DEBUG IMAGE
        self.wait_for_key = False # DEBUG KEY