class VideoOption:
    def __init__(self, folder_results='',inferenceParams={}, view_img=False, keep_resulting_video=False, debug_mode=False, save_img_bbox=True, weights='yolov7.pt', model_version='yolov10', compress_video=True, save_all_images=False,wait_for_key=False,show_config=False):
        self.weights = weights
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = '0'
        self.classes = [0]
        self.save_txt = False
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
        self.inferenceParams = inferenceParams
        self.save_img_bbox = save_img_bbox # Save every bounding box
        self.save_all_images = save_all_images
        self.debug_mode = debug_mode # DEBUG Show tracker config
        self.show_config = show_config # DEBUG Show tracker config
        self.keep_resulting_video = keep_resulting_video # DEBUG MODE True for not saving (production mode) False for saving (debug mode)
        self.compress_video = compress_video # DEBUG MODE (Being able to see the video without compression)
        self.view_img = view_img # DEBUG MODE (Show footage running in real time)
        self.wait_for_key = wait_for_key # DEBUG MODE (Wait for key press to continue)