class VideoOption:
    def __init__(self, folder_results='',view_img=False, keep_resulting_video=False, save_img_bbox=True, weights='yolov7.pt', model_version='yolov10', compress_video=True, save_all_images=False,tracker_selection='sort',bbox_centroid=None):
        self.weights = weights
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = '0'
        self.classes = [0]
        self.save_txt = False
        self.tracker_selection = tracker_selection
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
        self.save_img_bbox = save_img_bbox # Save every bounding box
        self.save_all_images = save_all_images
        self.show_config = True # DEBUG Show tracker config
        self.bbox_centroid = bbox_centroid # BOTTOM | TOP
        self.keep_resulting_video = keep_resulting_video # DEBUG MODE True for not saving (production mode) False for saving (debug mode)
        self.compress_video = compress_video # DEBUG MODE (Being able to see the video without compression)
        self.view_img = view_img # DEBUG MODE (Show footage running in real time)
        self.wait_for_key = False # DEBUG MODE (Wait for key press to continue)