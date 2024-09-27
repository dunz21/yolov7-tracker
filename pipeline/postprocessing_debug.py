from utils.draw_tools import filter_detections_inside_polygon,draw_polygon_interested_area,draw_boxes_entrance_exit,draw_configs,draw_boxes,distance_to_bbox_bottom_line,calculate_overlap,filter_model_detector_output,process_video_afterwards_for_debug


if __name__ == '__main__':
    
    base_path = '/home/diego/mydrive/results/7/22/1/lippi_costanera_entrada_20240916_1000/'
    video_path = f"{base_path}lippi_costanera_entrada_20240916_1000_COMPRESSED.mkv"
    csv_box_name = f"{base_path}lippi_costanera_entrada_20240916_1000_bbox.csv"
    entrance_line=entrance_line = [(619,1077), (402,730)]
    
    process_video_afterwards_for_debug(video_path, csv_box_name, entrance_line_in_out=entrance_line, view_img=False, wait_for_key=False)
    