import os
import cv2
import time
import torch
import logging
from pathlib import Path
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_log, \
    increment_path
from utils.torch_utils import select_device, load_classifier, \
    time_synchronized, TracedModel
from utils.download_weights import download
from intersect_ import *
# For SORT tracking
from sort import *
import time
from utils.draw_tools import filter_detections_inside_polygon,draw_polygon_interested_area,draw_boxes_entrance_exit,draw_configs,draw_boxes,distance_to_bbox_bottom_line,calculate_overlap,filter_model_detector_output
from utils.PersonImage import PersonImage
from utils.bytetrack.byte_tracker import BYTETracker
from utils.smile_track.mc_SMILEtrack import SMILEtrack
from utils.bytetrack.byte_tracker_adaptive import BYTETrackerAdaptive
from reid.BoundingBox import BoundingBox
from types import SimpleNamespace
from utils.compress_video import compress_and_replace_video
# from IPython import embed
import ast
from pipeline.main import process_pipeline_mini
from reid.VideoData import VideoData
from reid.VideoOption import VideoOption
from reid.VideoPipeline import VideoPipeline
from ultralytics import YOLOv10
from ultralytics.models import YOLO
# from dotenv import load_dotenv

def detect(video_data: VideoData, video_option: VideoOption, progress_callback=None, progress_interval=2) -> VideoPipeline:
    weights, view_img, save_txt, imgsz, trace, wait_for_key, save_bbox_dim, save_with_object_id = video_option.weights, video_option.view_img, video_option.save_txt, video_option.img_size, not video_option.no_trace, video_option.wait_for_key, video_option.save_bbox_dim, video_option.save_with_object_id
    # save_img = not video_option.nosave

    # .... Initialize SORT ....
    sort_max_age = 50
    sort_min_hits = 2
    sort_iou_thresh = 0.3
    
    obj = SimpleNamespace()
    obj.track_thresh = 0.5 ### BYTETRACK Default 0.5
    obj.match_thresh = 0.6 ### BYTETRACK Default 0.8
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
    
    if video_option.tracker_selection == 'sort':
        tracker_reid = Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)
    elif video_option.tracker_selection == 'smiletrack':
        tracker_reid = SMILEtrack(obj, frame_rate=15)
    elif video_option.tracker_selection == 'bytetrack':
        tracker_reid = BYTETracker(obj, frame_rate=15)
    elif video_option.tracker_selection == 'bytetrack_adaptive':
        tracker_reid = BYTETrackerAdaptive(obj, frame_rate=15)
    else:
        raise ValueError(f"Tracker selection {video_option.tracker_selection} not recognized")
        
    # .........................
    PersonImage.clear_instances()

    # video_option.name = video_data['folder_img']
    # Directories
    save_dir = Path(increment_path(Path(video_option.project) / video_data.name, exist_ok=video_option.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    os.chmod(save_dir, 0o775)
    # Initialize
    set_log(f"{save_dir}/app.log")
    logger = logging.getLogger(__name__)
    logger.info(vars(video_data))
    logger.info(vars(video_option))
    
    device = select_device(video_option.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    if video_option.model_version == 'yolov10':
        model = YOLOv10(weights)
        stride = 32  # Default stride for YOLOv10
    else:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        model = model.to(device)
        if half:
            model.half()  # to FP16
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, video_option.img_size)
        if half:
            model.half()  # to FP16


    imgsz = check_img_size(imgsz, s=stride)


    # Set Dataloader
    vid_path, vid_writer = None, None

    if 'rtsp' in video_data.source:
        dataset = LoadWebcam(pipe=video_data.source)
    else: 
        dataset = LoadImages(video_data.source, img_size=imgsz, stride=stride, print_info=True)
        
    # Check if the first file is a video or image
    vid_cap = None
    if dataset.video_flag[0]:
        vid_cap = dataset.cap
        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total_frames = dataset.nf  # number of images if no videos

    progress_step = int(total_frames * (progress_interval / 100))  # Calculate frames per progress update

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    t100 = time.time()
    current_frame = 0  # Initialize current frame counter

    for frame_index, (path, img, im0s, vid_cap, frame, valid_frame_iteration) in enumerate(dataset):
        if not valid_frame_iteration:
            continue
        save_dir_str = str(save_dir)
        folder_name_imgs = f"{save_dir_str}/imgs"
        csv_box_name = f"{save_dir_str}/{video_data.name}_bbox.csv"
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

        if video_option.model_version == 'yolov7':
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=video_option.augment)[0]
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=video_option.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            results = non_max_suppression(pred, video_option.conf_thres, video_option.iou_thres, classes=video_option.classes, agnostic=video_option.agnostic_nms)
            t3 = time_synchronized()
            
        if video_option.model_version == 'yolov10':
            # Inference
            t1 = time_synchronized()
            results = model(img, verbose=False, conf=video_option.conf_thres, iou=video_option.iou_thres)
            t2 = time_synchronized()
            t3 = time_synchronized()
            



        original_image = im0s.copy()
        if video_option.show_config:
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
                "bbox_centroid":video_option.bbox_centroid,
                }
            draw_configs(im0s,info,scale=im0s.shape[0])
        draw_polygon_interested_area(frame=im0s,polygon_pts=video_data.polygon_area)
        draw_boxes_entrance_exit(image=im0s,polygon_in=video_data.polygons_in,polygon_out=video_data.polygons_out)

        
        if tracker_reid.__class__.__name__ == 'Sort':
            trackers = tracker_reid.getTrackers()
            if len(trackers) > 0:
                for tracker in trackers:
                    if tracker.history.__len__() == sort_max_age:
                        id = tracker.id + 1
                        PersonImage.save(id=id, folder_name=folder_name_imgs, csv_box_name=csv_box_name,polygons_list=[video_data.polygons_in, video_data.polygons_out],FPS=FPS,save_img=video_option.save_img_bbox,save_all=video_option.save_all_images,bbox_centroid=video_option.bbox_centroid)
                        PersonImage.delete_instance(id)
        else :
            if tracker_reid.removed_stracks:
                unique_ids = set(val.track_id for val in tracker_reid.removed_stracks[-20:])
                for id in unique_ids:
                    remove_track_exists_in_tracker = any(val.track_id == id for val in tracker_reid.tracked_stracks)
                    if not remove_track_exists_in_tracker and PersonImage.get_instance(id):
                        PersonImage.save(id=id, folder_name=folder_name_imgs, csv_box_name=csv_box_name,polygons_list=[video_data.polygons_in, video_data.polygons_out],FPS=FPS,save_img=video_option.save_img_bbox,save_all=video_option.save_all_images,bbox_centroid=video_option.bbox_centroid)
                        PersonImage.delete_instance(id)
            
        # Process detections
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        
        
        for i, det in enumerate(results):  # detections per image
            if video_option.model_version == 'yolov10':
                det = det.boxes.data.clone()  # get box data
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                det_np = det.cpu().numpy()
                det_np = filter_detections_inside_polygon(detections=det_np,polygon_pts=video_data.polygon_area, bbox_complete_match=video_option.bbox_centroid is not None)
                det_np = filter_model_detector_output(yolo_output=det_np, specific_area_coords=video_data.filter_area)
                #if len(det_np) == 0:
                #   continue #Esto no permite que el Tracker se actualice, y matar los remove
                
                box_detection = [np.hstack([d[:4].astype(int),f"{d[4]:.2f}",0]) for d in det_np]
                draw_boxes(img=im0, bbox=box_detection, extra_info=None,color=(255,0,0),position='Bottom')
                
                # ..................USEa TRACK FUNCTION....................
                # pass an empty array to sort
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                # detections = det_np.cpu().detach().numpy()
                for x1, y1, x2, y2, conf, detclass in det_np:
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
                    
                    extra_info[id_tracker]['distance'] = distance_to_bbox_bottom_line(line=video_data.polygons_in[:2],bbox=box[:4])
                    for other_box in bbox_id:
                        if id_tracker != other_box[4]:
                            extra_info[id_tracker]['overlap'] += calculate_overlap(box[:4].astype(int), other_box[:4].astype(int))
                    
                    
                    objBbox = BoundingBox(
                        # img_frame=None, #for ram saving
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

        if progress_callback and frame_index % progress_step == 0:
            progress_callback(frame_index / total_frames * 100)
            logger.info(f"Progress: {frame_index / total_frames * 100:.2f}%")
        # Conditional print
        if not progress_callback:
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS. Mem: {PersonImage.get_memory_usage():.0f}Mb NumInstances: {PersonImage._instances.__len__()}')
            
        
        
        if frame % 1000 == 0:
            logger.info(f"1000 frames took {t100 - time.time()} seconds")
            t100 = time.time()
            
            
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
        
        # if not wait_for_key or key != ord('b'):  # Increment current frame unless 'b' is pressed or not waiting for key
        #     current_frame += 1


        # Save results (image with detections)
        if video_option.keep_resulting_video:
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
                    fps, w, h = 15, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)
        

    print(f'Done. ({time.time() - t0:.3f}s)')
    logger.info(f'Done. ({time.time() - t0:.3f}s)')
    
    if video_option.keep_resulting_video:
        vid_writer.release()
        if video_option.compress_video:
            compress_and_replace_video(save_path)
    
    return VideoPipeline(csv_box_name, save_path, folder_name_imgs, save_dir_str)