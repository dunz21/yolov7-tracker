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
from reid.BoundingBox import BoundingBox


from math import sqrt

DATA = [
    {
        'name' : "conce",
        'source' : "/home/diego/Documents/Footage/conce_HALF.mp4",
        'description' : "Video de Conce",
        'folder_img' : "imgs_conce",
        'polygons_in' : np.array([[263, 865],[583, 637],[671, 686],[344, 948]], np.int32),
        'polygons_out' : np.array([[202, 794],[508, 608],[583, 637],[263, 865]], np.int32),
        'polygon_area' : np.array([[0,1080],[0,600],[510,500],[593,523],[603,635],[632,653],[738,588],[756,860],[587,1080]], np.int32),
    },
    {
        'name' : "santos_dumont",
        'source' : "/home/diego/Documents/Footage/SANTOS LAN_ch6.mp4",
        'description' : "Video de Santos Dumont",
        'folder_img' : "imgs_santos_dumont",
        'polygons_in' : np.array([[865, 532],[1117,570],[1115,635],[831,581]], np.int32),
        'polygons_out' : np.array([[918,498],[1112,522],[1114,570],[865,527]], np.int32),
        'polygon_area' : np.array([[710,511],[712,650],[1119,757],[1206,562],[1179,378],[731,325]], np.int32),
    },
    {
        'name' : "santos_dumont_split",
        'source' : "/home/diego/Documents/Footage/TEST_FRAMES/",
        'description' : "Video de Santos Dumont",
        'folder_img' : "imgs_santos_split",
        'polygons_in' : np.array([[865, 532],[1117,570],[1115,635],[831,581]], np.int32),
        'polygons_out' : np.array([[918,498],[1112,522],[1114,570],[865,527]], np.int32),
        'polygon_area' : np.array([[710,511],[712,650],[1119,757],[1206,562],[1179,378],[731,325]], np.int32),
    },
    {
        'name' : "webcam",
        'source' : "rtsp://admin:OTWBMF@201.215.37.171:554/H.264",
        'description' : "Test IP",
        'folder_img' : "imgs_webcam",
        'polygons_in' : np.array([[591,515],[610,557],[735,515],[736,480],[707,488]], np.int32),
        'polygons_out' : np.array([[590,489],[701,464],[735,477],[591,511]], np.int32),
        'polygon_area' : np.array([[493,407],[569,700],[937,561],[826,316]], np.int32),
    },
    {
        'name' : "conce_test",
        'source' : "/home/diego/Documents/Footage/conce_debug_3.mp4",
        'description' : "Video de Conce",
        'folder_img' : "imgs_conce_debug",
        'polygons_in' : np.array([[263, 865],[583, 637],[671, 686],[344, 948]], np.int32),
        'polygons_out' : np.array([[202, 794],[508, 608],[583, 637],[263, 865]], np.int32),
        'polygon_area' : np.array([[0,1080],[0,600],[510,500],[593,523],[603,635],[632,653],[738,588],[756,860],[587,1080]], np.int32),
    },
]

def draw_boxes(img, bbox, identities=None , offset=(0, 0),extra_info=None,color=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0

        label = str(id) + ":" + "person"
        if extra_info is not None:
            label += str(f"oc:{extra_info[id]['overlap']:.2f}")
            label += str(f"di:{extra_info[id]['distance']:.2f}")

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if color is None:
            color = (255, 0, 20)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

    return img

def calculate_overlap(rect1, rect2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both rectangles
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate union area
    union_area = area1 + area2 - intersection_area

    # Calculate the overlap percentage
    overlap = intersection_area / union_area

    return overlap

def distance_to_bbox_centroid(point, bbox):
    """
    Calculate the distance between a point and the center of the bottom edge of a bounding box (bbox).

    :param point: A tuple representing the point (x, y).
    :param bbox: A tuple representing the bounding box (x1, y1, x2, y2).
    :return: The Euclidean distance between the point and the center of the bottom edge of the bbox.
    """
    px, py = point
    x1, y1, x2, y2 = bbox

    # Calculate the center of the bottom edge of the bbox
    bottom_center_x = (x1 + x2) / 2
    bottom_center_y = y2

    # Calculate Euclidean distance from the point to the bottom center
    distance = sqrt((bottom_center_x - px) ** 2 + (bottom_center_y - py) ** 2)
    return distance

def detect(save_img=False,video_data=None):
    source, weights, view_img, save_txt, imgsz, trace, wait_for_key, save_bbox_dim, save_with_object_id = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.wait_for_key, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # .... Initialize SORT ....
    sort_max_age = 50
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    # .........................
    PersonImage.clear_instances()

    opt.name = video_data['folder_img']
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
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
        center_of_interested = np.mean(polygons_in_out[0][:2],axis=0).squeeze()


        #Puedo tener trackers y ninguna deteccion. Ya que pueden ser del pasado
        trackers = sort_tracker.getTrackers()
        if len(trackers) > 0:
            for tracker in trackers:
                if tracker.bbox_history.__len__() > 500: # en caso de que las personas se queden paradas
                    PersonImage.delete_instance(tracker.id + 1)

                if tracker.history.__len__() == sort_max_age:
                    PersonImage.delete_instance(tracker.id + 1)
                    continue
                if tracker.history.__len__() == 10:
                    PersonImage.save(id=tracker.id + 1,folder_name=video_data['folder_img'],csv_box_name=f"{video_data['name']}_bbox.csv")

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det = det.cpu().detach().numpy()
                det = filter_detections_inside_polygon(detections=det,polygon_pts=video_data['polygon_area'])

                # ..................USE TRACK FUNCTION....................
                # pass an empty array to sort
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                # detections = det.cpu().detach().numpy()
                for x1, y1, x2, y2, conf, detclass in det:
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2, conf, detclass])))

                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                # loop over tracks
                for track in tracks:
                    only_bboxes = [box[:4] for box in track.bbox_history]
                    new_person = PersonImage(id=track.id+1,list_images=[],history_deque=only_bboxes)
                    result = new_person.find_polygons_for_centroids(polygons_in_out)
                    direction_and_position = new_person.detect_pattern_change(result)

                    # UPDATE a la direccion para luego guardar con el tracker salga
                    if direction_and_position is not None:
                        direction = direction_and_position[0]
                        position = direction_and_position[1]
                        if position % 2 == 0:
                            new_person.direction = 'In' if direction == '10' else 'Out'
                
                # loop over detections, in the future we can use this to not loop over tracks
                for track_det in tracked_dets:
                    id = int(track_det[8])
                    conf = track_det[9]
                    bbox = [*track_det[:4].astype(int),conf]
                    x1, y1, x2, y2 = bbox[:4]
                    sub_frame = original_image[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                    distance_to_center = distance_to_bbox_centroid(center_of_interested,bbox[:4])


                    total_overlap_tracker = 0
                    for other_track_det in tracked_dets:
                        id_other_track = other_track_det[8]
                        if id != id_other_track:
                            total_overlap_tracker += calculate_overlap(bbox[:4], other_track_det[:4].astype(int))

                    new_person = PersonImage(id=id,list_images=[])
                    box = BoundingBox(
                        img_frame=sub_frame,
                        frame_number=getattr(dataset, 'total_frame_videos', 0) + frame,
                        bbox=bbox,
                        overlap=total_overlap_tracker,
                        distance_to_center=distance_to_center)
                    new_person.list_images.append(box)



            else:
                tracked_dets = sort_tracker.update()


        # draw boxes for visualization
        if len(tracked_dets) > 0:
            bbox_xyxy = tracked_dets[:, :4]
            # bbox_xyxy = [track.bbox_history[-1][:4] for track in sort_tracker.getTrackers() if len(track.history) == 0 and len(track.bbox_history) > 2]
            identities = tracked_dets[:, 8]
            

            #TODO: DEJARLO OPCIONAL CON HIPERPARAMETROS
            extra_info = {}
            for actual_track in tracked_dets:
                track_id = actual_track[8]
                if track_id not in extra_info:
                    extra_info[track_id] = {'overlap': 0}
                if 'distance' not in extra_info[track_id]:
                    extra_info[track_id]['distance'] = distance_to_bbox_centroid(center_of_interested, actual_track[:4])
                for other_track in tracked_dets:
                    if actual_track[8] != other_track[8]:
                        extra_info[track_id]['overlap'] += calculate_overlap(actual_track[:4], other_track[:4])

            # if (len(bbox_xyxy) == len(identities)):
            #     draw_boxes(img=im0, bbox=bbox_xyxy, identities=identities,extra_info=extra_info,color=(0,255,0))
            # else:
            #     bbox_xyxy = tracked_dets[:, :4]
            #     draw_boxes(img=im0, bbox=bbox_xyxy, identities=identities,extra_info=extra_info)
            draw_boxes(img=im0, bbox=bbox_xyxy, identities=identities,extra_info=extra_info)

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
    print(f'Done. ({time.time() - t0:.3f}s)')
    print([f"{t:.2f}" for t in time_for_each_100_frames])
    


class Options:
    def __init__(self):
        self.weights = 'yolov7.pt'
        # self.source = '/home/diego/Documents/Footage/santos10_min.mp4'
        self.source = '/home/diego/Documents/detectron2/mini_conce.mp4'
        # self.source = 'retail.mp4'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = '0'
        self.view_img = False # DEBUG
        self.save_txt = False
        self.save_conf = False
        self.nosave = False
        self.classes = [0]
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.project = 'runs/detect'
        self.name = 'diponti_sto_dumont'
        self.exist_ok = False
        self.no_trace = False
        self.wait_for_key = False # DEBUG
        self.save_bbox_dim = False
        self.save_with_object_id = False
        self.download = True


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
                video_data = DATA[0]
                detect(video_data=video_data)
                # getFinalScore(folder_name=video_data['folder_img'],solider_file=f"{video_data['name']}_solider_in-out.csv",silhoutte_file=f"{video_data['name']}_distance_cosine.csv",html_file=f"{video_data['name']}_cosine_match.html",distance_method="cosine")
                # getFinalScore(folder_name=video_data['folder_img'],solider_file=f"{video_data['name']}_solider_in-out.csv",silhoutte_file=f"{video_data['name']}_distance_kmeans.csv",html_file=f"{video_data['name']}_kmeans_match.html",distance_method="kmeans")
                # export_images_in_out_to_html(f"{video_data['name']}_distance_kmeans.csv",f"{video_data['name']}_solider_in-out.csv",video_data['folder_img'],f"{video_data['name']}_all_images.html")

            # except:
            #     print("Error")
            
            # try:
                video_data = DATA[1]
            #     detect(video_data=video_data)
                # getFinalScore(folder_name=video_data['folder_img'],solider_file=f"{video_data['name']}_solider_in-out.csv",silhoutte_file=f"{video_data['name']}_distance_cosine.csv",html_file=f"{video_data['name']}_cosine_match.html",distance_method="cosine")
                # getFinalScore(folder_name=video_data['folder_img'],solider_file=f"{video_data['name']}_solider_in-out.csv",silhoutte_file=f"{video_data['name']}_distance_kmeans.csv",html_file=f"{video_data['name']}_kmeans_match.html",distance_method="kmeans")
                # export_images_in_out_to_html(f"{video_data['name']}_distance_kmeans.csv",f"{video_data['name']}_solider_in-out.csv",video_data['folder_img'],f"{video_data['name']}_all_images.html")
            # except:
            #     print("Error")
            
            
                
                
                