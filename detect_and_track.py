import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn
from collections import Counter
from collections import deque
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
    time_synchronized, TracedModel
from utils.download_weights import download
from intersect_ import *
# For SORT tracking
from sort import *
import time
from utils.draw_tools import filter_detections_inside_polygon,draw_polygon_interested_area,draw_boxes_entrance_exit,draw_configs,draw_boxes,distance_to_bbox_bottom_line,calculate_overlap
from utils.PersonImage import PersonImage
from utils.bytetrack.byte_tracker import BYTETracker
from utils.smile_track.mc_SMILEtrack import SMILEtrack
from utils.bytetrack.byte_tracker_adaptive import BYTETrackerAdaptive
from utils.video_data import get_video_data
from reid.BoundingBox import BoundingBox
from shapely.geometry import LineString, Point
from types import SimpleNamespace
from datetime import datetime
from pipeline.vit_pipeline import get_features_from_model
from pipeline.re_ranking import complete_re_ranking
from utils.compress_video import compress_and_replace_video
from IPython import embed
import ast
from pipeline.main import process_pipeline
# from dotenv import load_dotenv


def detect(save_img=False,video_data=None):
    weights, view_img, show_config, save_txt, imgsz, trace, wait_for_key, save_bbox_dim, save_with_object_id = opt.weights, opt.view_img, opt.show_config, opt.save_txt, opt.img_size, not opt.no_trace, opt.wait_for_key, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave

    # .... Initialize SORT ....
    sort_max_age = 50
    sort_min_hits = 2
    sort_iou_thresh = 0.3
    
    obj = SimpleNamespace()
    obj.track_thresh = 0.5 ### BYTETRACK Default 0.5
    obj.match_thresh = 0.8 ### BYTETRACK Default 0.8
    obj.track_high_thresh = 0.2 ### SMILE TRACK ONLY Default 0.6
    obj.track_low_thresh = 0.1 ### SMILE TRACK ONLY Default 0.1
    obj.new_track_thresh = 0.2 ### SMILE TRACK ONLY Default 0.7
    obj.proximity_thresh = 0.5 ### SMILE TRACK ONLY Default 0.5
    obj.appearance_thresh = 0.25 ### SMILE TRACK ONLY Default 0.25
    obj.with_reid = False ### SMILE TRACK ONLY Default False
    obj.cmc_method = 'sparseOptFlow' ### SMILE TRACK ONLY Default orb|sift|ecc|sparseOptFlow|file|None
    obj.track_buffer = 50
    obj.mot20 = False
    obj.aspect_ratio_thresh = 1.6   
    obj.min_box_area = 10
    ### SORT TRACKER###
    obj.sort_iou_thresh = sort_iou_thresh
    
    
    # tracker_reid = SMILEtrack(obj, frame_rate=15)
    tracker_reid = BYTETracker(obj, frame_rate=15)
    # tracker_reid = BYTETrackerAdaptive(obj, frame_rate=15)
    # tracker_reid = Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)
    # .........................
    PersonImage.clear_instances()

    # opt.name = video_data['folder_img']
    # Directories
    formatted_date = datetime.now().strftime('%Y_%m_%d')
    folder_name = f"{formatted_date}_{video_data['name']}"
    save_dir = Path(increment_path(Path(opt.project) / folder_name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    os.chmod(save_dir, 0o775)
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None

    if 'rtsp' in video_data['source']:
        dataset = LoadWebcam(pipe=video_data['source'])
    else: 
        dataset = LoadImages(video_data['source'], img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    t100 = time.time()
    time_for_each_100_frames = []
    current_frame = 0  # Initialize current frame counter

    for path, img, im0s, vid_cap, frame, valid_frame_iteration in dataset:
        if not valid_frame_iteration:
            continue
        save_dir_str = str(save_dir)
        folder_name = f"{save_dir_str}/{video_data['folder_img']}"
        csv_box_name = f"{save_dir_str}/{video_data['name']}_bbox.csv"
        # FPS = vid_cap.get(cv2.CAP_PROP_FPS)
        FPS = 15
        # if width == 0:
        #     total_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     total_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()


        original_image = im0s.copy()
        if show_config:
            info = {
                "sort_iou_thresh":obj.sort_iou_thresh,
                "track_thresh":obj.track_thresh,
                "match_thresh":obj.match_thresh,
                "track_buffer":obj.track_buffer,
                "track_high_thresh":obj.track_high_thresh,
                "track_low_thresh":obj.track_low_thresh,
                "new_track_thresh":obj.new_track_thresh,
                "proximity_thresh":obj.proximity_thresh,
                "appearance_thresh":obj.appearance_thresh,
                "cmc_method":obj.cmc_method,
                "aspect_ratio_thresh":obj.aspect_ratio_thresh,
                "min_box_area":obj.min_box_area,
                "tracker" : tracker_reid.__class__.__name__,
                "weights":weights.split("/")[-1],
                }
            draw_configs(im0s,info)
        draw_polygon_interested_area(frame=im0s,polygon_pts=video_data['polygon_area'])
        draw_boxes_entrance_exit(image=im0s,polygon_in=video_data['polygons_in'],polygon_out=video_data['polygons_out'])

        
        if tracker_reid.__class__.__name__ == 'Sort':
            trackers = tracker_reid.getTrackers()
            if len(trackers) > 0:
                for tracker in trackers:
                    if tracker.history.__len__() == sort_max_age:
                        id = tracker.id + 1
                        PersonImage.save(id=id, folder_name=folder_name, csv_box_name=csv_box_name,polygons_list=[video_data['polygons_in'], video_data['polygons_out']],FPS=FPS)
                        PersonImage.delete_instance(id)
        else :
            if tracker_reid.removed_stracks:
                unique_ids = set(val.track_id for val in tracker_reid.removed_stracks[-20:])
                for id in unique_ids:
                    remove_track_exists_in_tracker = any(val.track_id == id for val in tracker_reid.tracked_stracks)
                    if not remove_track_exists_in_tracker and PersonImage.get_instance(id):
                        PersonImage.save(id=id, folder_name=folder_name, csv_box_name=csv_box_name,polygons_list=[video_data['polygons_in'], video_data['polygons_out']],FPS=FPS)
                        PersonImage.delete_instance(id)
            
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det = det.cpu().detach().numpy()
                det = filter_detections_inside_polygon(detections=det,polygon_pts=video_data['polygon_area'])
                # if len(det) == 0:
                #     continue #Esto no permite que el Tracker se actualice, y matar los remove
                box_detection = [np.hstack([d[:4].astype(int),f"{d[4]:.2f}",0]) for d in det]
                draw_boxes(img=im0, bbox=box_detection, extra_info=None,color=(255,0,0),position='Bottom')
                
                # ..................USEa TRACK FUNCTION....................
                # pass an empty array to sort
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                # detections = det.cpu().detach().numpy()
                for x1, y1, x2, y2, conf, detclass in det:
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2, conf, detclass])))
                    
                #### BYTETRACK
                online_targets = tracker_reid.update(dets_to_sort.copy(), im0) if tracker_reid.__class__.__name__ == 'SMILEtrack' else tracker_reid.update(dets_to_sort.copy())
                bbox_id = [np.hstack([track[0:4],track[-2],track[-1]]) for track in online_targets] if tracker_reid.__class__.__name__ == 'Sort' else [np.hstack([track.tlbr,track.track_id,track.score]) for track in online_targets]
                extra_info = {}
                for box in bbox_id:
                    x1, y1, x2, y2, id_tracker, score = box
                    extra_info[id_tracker] = {'overlap': 0, 'distance' : 0}
                    
                    id_tracker = int(id_tracker)
                    if (PersonImage.get_instance(id_tracker) == None):
                        new_person = PersonImage(id=id_tracker,list_images=[],history_deque=[])
                    else:
                        new_person = PersonImage.get_instance(id_tracker)
                        
                    sub_frame = original_image[max(0,int(y1)):max(0,int(y2)), max(0,int(x1)):max(0,int(x2))]
                    
                    extra_info[id_tracker]['distance'] = distance_to_bbox_bottom_line(line=video_data['polygons_in'][:2],bbox=box[:4])
                    for other_box in bbox_id:
                        if id_tracker != other_box[4]:
                            extra_info[id_tracker]['overlap'] += calculate_overlap(box[:4].astype(int), other_box[:4].astype(int))
                    
                    
                    objBbox = BoundingBox(
                        img_frame=sub_frame if frame % 3 == 0 else None, #for ram saving
                        frame_number=getattr(dataset, 'total_frame_videos', 0) + frame,
                        bbox=[*box[:4].astype(int),score],
                        overlap=extra_info[id_tracker]['overlap'],
                        distance_to_center=extra_info[id_tracker]['distance'])
                    
                    new_person.list_images.append(objBbox)
                    new_person.history_deque.append(box[:4])
                    if(new_person.history_deque.__len__() > 500):
                        with open(f'{str(save_dir)}/tracker.txt', 'a') as log_file:
                            log_file.write(f"DELETE: {id_tracker} History: {new_person.history_deque.__len__()}  \n")
                        PersonImage.delete_instance(id_tracker)
                
                draw_boxes(img=im0, bbox=bbox_id, extra_info=extra_info,color=(0,0,255),position='Top')

            
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS. Mem: {PersonImage.get_memory_usage():.0f}Mb NumInstances: {PersonImage._instances.__len__()}')

        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            if wait_for_key:
                key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press if wait_for_key is True
            else:
                key = cv2.waitKey(1) & 0xFF  # Use a short wait time and proceed if wait_for_key is False
            
            if key == 27:  # If 'ESC' is pressed, break the loop
                break
            elif key == ord('b'):  # If 'b' is pressed, move back one frame
                current_frame = max(0, current_frame - 1)  # Ensure current_frame does not go below 0
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                continue
            # Add your other key handling here...
        
        if not wait_for_key or key != ord('b'):  # Increment current frame unless 'b' is pressed or not waiting for key
            current_frame += 1


        # Save results (image with detections)
        if save_img:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    # fps = vid_cap.get(cv2.CAP_PROP_FPS) #TODO: Check this 
                    fps = 15
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)
            
        # if frame % 100 == 0:
        #     time_for_each_100_frames.append(time.time() - t100)
        #     t100 = time.time()
            
    # db_base_path = f"{csv_box_name}.db"
    ### Limpieza switch ID 
    # convert_csv_to_sqlite(csv_file_path=f"{csv_box_name}.csv", db_file_path=db_base_path, table_name='bbox_raw')
    # switch_id_corrector_pipeline(db_path=db_base_path, base_folder_path=folder_name,weights='model_weights.pth',model_name='solider')
    # prepare_data_img_selection(db_path=db_base_path, origin_table="bbox_raw", k_folds=4, n_images=5, new_table_name="bbox_img_selection")
    # predict_img_selection(db_file_path=db_base_path, model_weights_path='mini_models/results/image_selection_model.pkl')
    # clean_img_folder_top_k(db_file_path=db_base_path, base_folder_images=folder_name, dest_folder_results=f"{folder_name}_top4", k_fold=4, threshold=0.9)
    # features = get_features_from_model(model_name='solider', folder_path=f"{folder_name}_top4", weights='model_weights.pth', db_path=db_base_path)
    # complete_re_ranking(features,n_images=8,max_number_back_to_compare=57,K1=8,K2=3,LAMBDA=0,db_path=db_base_path)

    
    # with open(f'{str(save_dir)}/tracker.txt', 'a') as log_file:
    #     formatted_times = [f"{t:.2f}" for t in time_for_each_100_frames]
    #     log_file.write(f"Time for each 100 frames: {formatted_times} \n")
    print(f'Done. ({time.time() - t0:.3f}s)')
    with open(f'{save_dir_str}/tracker.txt', 'a') as log_file:
        log_file.write(f"{time.time() - t0:.3f}[s] \n")
    
    if save_img:
        vid_writer.release()
        compress_and_replace_video(save_path,encoder='libx264')
    
    return csv_box_name,save_path, folder_name

def load_video_data():
    if 'ENV_FILE' in os.environ:
        video_dir = os.getenv('VIDEO_DIR_CONTAINER', '/app/videos')  # Always use the Docker container's video directory
        video_file = os.getenv('VIDEO_FILE', 'video.mp4')
        video_path = os.path.join(video_dir, video_file)
        print(f"Video path: {video_path}")
        video_data = {
            'name': os.getenv('name'),
            'source': video_path,
            'description': os.getenv('description'),
            'folder_img': os.getenv('folder_img'),
            'polygons_in': np.array(eval(os.getenv('polygons_in')), np.int32),
            'polygons_out': np.array(eval(os.getenv('polygons_out')), np.int32),
            'polygon_area': np.array(eval(os.getenv('polygon_area')), np.int32),
            'client_id': os.getenv('CLIENT_ID'),
            'store_id': os.getenv('STORE_ID'),
            'video_date': os.getenv('VIDEO_DATE'),
            'start_time_video': os.getenv('START_TIME_VIDEO'),
            'frame_rate_video': os.getenv('FRAME_RATE_VIDEO')
        }
    else:
        video_data = {
            'name': "santos_dumont_debug",
            'source': "/home/diego/Documents/Footage/dumont_debug1.mp4",
            'description': "Video de Santos Dumont",
            'folder_img': "imgs_santos_dumont_debug",
            'polygons_in': np.array([[865, 510],[1117,550],[1115,595],[831,541]], np.int32),
            'polygons_out': np.array([[894, 480],[1118,510],[1117,550],[865,510]], np.int32),
            'polygon_area': np.array([[731,325],[1179,378],[1206,562],[1119,1050],[442,850],[710,511]], np.int32),
            'client_id': 1,
            'store_id': 3,
            'video_date': "2021-09-01",
            'start_time_video': '10:00:00',
            'frame_rate_video': 15
        }
    return video_data

class Options:
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


if __name__ == '__main__':
    opt = Options()
    print(opt.__dict__)

    if opt.download and not os.path.exists(''.join(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        # load_dotenv()
        video_data = load_video_data()
        # csv,video,img_folder = detect(video_data=video_data)
        csv,video,img_folder = 'runs/detect/2024_05_25_tobalaba_docker/tobalaba_docker_bbox.csv' , 'runs/detect/2024_05_25_tobalaba_docker/tobalaba_2024-05-21.mp4','runs/detect/2024_05_25_tobalaba_docker/imgs_diponti_tobalaba'
        process_pipeline(csv_box_name=csv, video_path=video, img_folder_name=img_folder,client_id=video_data['client_id'],store_id=video_data['store_id'],video_date=video_data['video_date'],start_time_video=video_data['start_time_video'],frame_rate=video_data['frame_rate_video'])
        
        
        
        

            
            
                
                
                