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
from utils.draw_tools import filter_detections_inside_polygon,draw_polygon_interested_area,draw_boxes_entrance_exit
from utils.PersonImage import PersonImage
from utils.bytetrack.byte_tracker import BYTETracker
from utils.video_data import get_video_data
from reid.BoundingBox import BoundingBox
from shapely.geometry import LineString, Point
from types import SimpleNamespace
from datetime import datetime
from utils.tools import distance_to_bbox_bottom_line,calculate_overlap,draw_boxes,convert_csv_to_sqlite

def detect(save_img=False,video_data=None):
    weights, view_img, save_txt, imgsz, trace, wait_for_key, save_bbox_dim, save_with_object_id = opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.wait_for_key, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave

    # .... Initialize SORT ....
    sort_max_age = 50
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    obj = SimpleNamespace()
    obj.track_thresh = 0.5
    obj.track_buffer = 50
    obj.mot20 = False
    obj.match_thresh = 0.8
    obj.aspect_ratio_thresh = 1.6
    obj.min_box_area = 10
    bytetrack = BYTETracker(obj, frame_rate=15)
    # .........................
    PersonImage.clear_instances()

    # opt.name = video_data['folder_img']
    # Directories
    formatted_date = datetime.now().strftime('%Y_%m_%d')
    folder_name = f"{formatted_date}_{video_data['name']}"
    save_dir = Path(increment_path(Path(opt.project) / folder_name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

    width = 0
    height = 0
    time_for_each_100_frames = []
    results = []
    
    for path, img, im0s, vid_cap, frame in dataset:
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
        # draw_polygon_interested_area(frame=im0s,polygon_pts=video_data['polygon_area'])
        polygons_in_out = draw_boxes_entrance_exit(image=im0s,polygon_in=video_data['polygons_in'],polygon_out=video_data['polygons_out'])


        #Puedo tener trackers y ninguna deteccion. Ya que pueden ser del pasado
        trackers = sort_tracker.getTrackers()



        # if len(trackers) > 0:
        #     for tracker in trackers:
        #         if tracker.bbox_history.__len__() > 500: # en caso de que las personas se queden paradas no muera por ram
        #             PersonImage.delete_instance(tracker.id + 1)
        #         if tracker.history.__len__() == sort_max_age:
        #             PersonImage.delete_instance(tracker.id + 1)
        #             continue
        #         if tracker.history.__len__() == 10:
        #             PersonImage.save(id=tracker.id + 1,folder_name=video_data['folder_img'],csv_box_name=f"{video_data['name']}_bbox",polygons_list=[video_data['polygons_in'],video_data['polygons_out']])
                    
        if (bytetrack.removed_stracks.__len__() > 0):
            for id in np.unique(np.array([val.track_id for val in bytetrack.removed_stracks])):
                if (PersonImage.get_instance(id)):    
                    PersonImage.save(id=id,folder_name=f"{str(save_dir)}/{video_data['folder_img']}",csv_box_name=f"{str(save_dir)}/{video_data['name']}_bbox",polygons_list=[video_data['polygons_in'],video_data['polygons_out']])
                    PersonImage.delete_instance(id)
                    with open(f'{str(save_dir)}/tracker.txt', 'a') as log_file:
                        log_file.write(f"SAVED: {id} \n")
            
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det = det.cpu().detach().numpy()
                det = filter_detections_inside_polygon(detections=det,polygon_pts=video_data['polygon_area'])


               


                # ..................USEa TRACK FUNCTION....................
                # pass an empty array to sort
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                # detections = det.cpu().detach().numpy()
                for x1, y1, x2, y2, conf, detclass in det:
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2, conf, detclass])))
                #### BYTETRACK
                online_targets = bytetrack.update(dets_to_sort.copy(), [int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))], img.shape[2:])
                bbox_id = [np.hstack([track.tlbr,track.track_id,track.score]) for track in online_targets]
                for box in bbox_id:
                    x1, y1, x2, y2, id, score = box
                    id = int(id)
                    if (PersonImage.get_instance(id) == None):
                        new_person = PersonImage(id=id,list_images=[],history_deque=[])
                    else:
                        new_person = PersonImage.get_instance(id)
                        
                    sub_frame = original_image[max(0,int(y1)):max(0,int(y2)), max(0,int(x1)):max(0,int(x2))]
                    distance_to_center = distance_to_bbox_bottom_line(line=video_data['polygons_in'][:2],bbox=box[:4])

                    total_over_lap = 0
                    for other_box in bbox_id:
                        if box[4] != other_box[4]:
                            total_over_lap += calculate_overlap(box[:4].astype(int), other_box[:4].astype(int))
                    
                    objBbox = BoundingBox(
                        img_frame=sub_frame,
                        frame_number=getattr(dataset, 'total_frame_videos', 0) + frame,
                        bbox=[*box[:4].astype(int),score],
                        overlap=total_over_lap,
                        distance_to_center=distance_to_center)
                    
                    new_person.list_images.append(objBbox)
                    new_person.history_deque.append(box[:4])
                    if(new_person.history_deque.__len__() > 500):
                        with open(f'{str(save_dir)}/tracker.txt', 'a') as log_file:
                            log_file.write(f"DELETE: {id} History: {new_person.history_deque.__len__()}  \n")
                        PersonImage.delete_instance(id)
                
                draw_boxes(img=im0, bbox=bbox_id, extra_info=None,color=(0,0,255),position='Top')

                
                
                
                            
                # Run SORT
                # tracked_dets = sort_tracker.update(dets_to_sort)
                # tracks = sort_tracker.getTrackers()

                # # loop over tracks
                # for track in tracks:
                #     only_bboxes = [box[:4] for box in track.bbox_history]
                #     new_person = PersonImage(id=track.id+1,list_images=[],history_deque=only_bboxes)
                
                # # loop over detections, in the future we can use this to not loop over tracks
                # for track_det in tracked_dets:
                #     id = int(track_det[8])
                #     conf = track_det[9]
                #     bbox = [*track_det[:4].astype(int),conf]
                #     x1, y1, x2, y2 = bbox[:4]
                #     sub_frame = original_image[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                #     distance_to_center = distance_to_bbox_bottom_line(line=video_data['polygons_in'][:2],bbox=bbox[:4])


                #     total_overlap_tracker = 0
                #     for other_track_det in tracked_dets:
                #         id_other_track = other_track_det[8]
                #         if id != id_other_track:
                #             total_overlap_tracker += calculate_overlap(bbox[:4], other_track_det[:4].astype(int))

                #     new_person = PersonImage.get_instance(id)
                #     box = BoundingBox(
                #         img_frame=sub_frame,
                #         frame_number=getattr(dataset, 'total_frame_videos', 0) + frame,
                #         bbox=bbox,
                #         overlap=total_overlap_tracker,
                #         distance_to_center=distance_to_center)
                #     new_person.list_images.append(box)



            else:
                tracked_dets = sort_tracker.update()


        # draw boxes for visualization
        # if len(tracked_dets) > 0:
        #     bbox_xyxy = tracked_dets[:, :4]
        #     # bbox_xyxy = [track.bbox_history[-1][:4] for track in sort_tracker.getTrackers() if len(track.history) == 0 and len(track.bbox_history) > 2]
        #     identities = tracked_dets[:, 8]
            

            #TODO: DEJARLO OPCIONAL CON HIPERPARAMETROS
            # extra_info = {}
            # for actual_track in tracked_dets:
            #     track_id = actual_track[8]
            #     if track_id not in extra_info:
            #         extra_info[track_id] = {'overlap': 0}
            #     if 'distance' not in extra_info[track_id]:
            #         extra_info[track_id]['distance'] = distance_to_bbox_bottom_line(line=video_data['polygons_in'][:2],bbox=bbox[:4])
            #     for other_track in tracked_dets:
            #         if actual_track[8] != other_track[8]:
            #             extra_info[track_id]['overlap'] += calculate_overlap(actual_track[:4], other_track[:4])

            # if (len(bbox_xyxy) == len(identities)):
            #     draw_boxes(img=im0, bbox=bbox_xyxy, identities=identities,extra_info=extra_info,color=(0,255,0))
            # else:
            #     bbox_xyxy = tracked_dets[:, :4]
            #     draw_boxes(img=im0, bbox=bbox_xyxy, identities=identities,extra_info=extra_info)
            # draw_boxes(img=im0, bbox=bbox_xyxy, identities=identities,extra_info=None)

        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS. Mem: {PersonImage.get_memory_usage():.0f}Mb NumInstances: {PersonImage._instances.__len__()}')

        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            if wait_for_key:
                key = cv2.waitKey(0)
                if key == 27: # If 'ESC' is pressed, break the loop
                    raise StopIteration
            if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:  # q to quit
                cv2.destroyAllWindows()
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)
    convert_csv_to_sqlite(csv_file_path=f"{str(save_dir)}/{video_data['name']}_bbox.csv",db_file_path=f"{str(save_dir)}/{video_data['name']}_bbox.db",table_name='bbox_raw')
    print(f'Done. ({time.time() - t0:.3f}s)')
    print([f"{t:.2f}" for t in time_for_each_100_frames])
    


class Options:
    def __init__(self):
        self.weights = 'yolov7.pt'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = '0'
        self.save_txt = False
        self.save_conf = False
        self.nosave = False
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
        self.view_img = False # DEBUG IMAGE
        self.wait_for_key = False # DEBUG KEY


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true',
                        help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',
                        help='not download model weights if already exist')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='object_tracking',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true',
                        help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true',
                        help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true',
                        help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true',
                        help='save results with object id to *.txt')

    parser.set_defaults(download=True)
# opt = parser.parse_args() OLD
    opt = Options()
    print(opt.__dict__)
    # check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(''.join(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            # try:
                DATA = get_video_data()
                video_data = next((final for final in DATA if final['name'] == 'conce_test'), None)
                detect(video_data=video_data)
                # getFinalScore(folder_name=video_data['folder_img'],solider_file=f"{video_data['name']}_solider_in-out.csv",silhoutte_file=f"{video_data['name']}_distance_cosine.csv",html_file=f"{video_data['name']}_cosine_match.html",distance_method="cosine")
                # getFinalScore(folder_name=video_data['folder_img'],solider_file=f"{video_data['name']}_solider_in-out.csv",silhoutte_file=f"{video_data['name']}_distance_kmeans.csv",html_file=f"{video_data['name']}_kmeans_match.html",distance_method="kmeans")
                # export_images_in_out_to_html(f"{video_data['name']}_distance_kmeans.csv",f"{video_data['name']}_solider_in-out.csv",video_data['folder_img'],f"{video_data['name']}_all_images.html")

            # except:
            #     print("Error")
            
            # try:
                # video_data = DATA[4]
            #     detect(video_data=video_data)
                # getFinalScore(folder_name=video_data['folder_img'],solider_file=f"{video_data['name']}_solider_in-out.csv",silhoutte_file=f"{video_data['name']}_distance_cosine.csv",html_file=f"{video_data['name']}_cosine_match.html",distance_method="cosine")
                # getFinalScore(folder_name=video_data['folder_img'],solider_file=f"{video_data['name']}_solider_in-out.csv",silhoutte_file=f"{video_data['name']}_distance_kmeans.csv",html_file=f"{video_data['name']}_kmeans_match.html",distance_method="kmeans")
                # export_images_in_out_to_html(f"{video_data['name']}_distance_kmeans.csv",f"{video_data['name']}_solider_in-out.csv",video_data['folder_img'],f"{video_data['name']}_all_images.html")
            # except:
            #     print("Error")
            
            
                
                
                