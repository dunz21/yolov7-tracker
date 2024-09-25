from utils.draw_tools import filter_detections_inside_polygon,draw_polygon_interested_area,draw_boxes_entrance_exit,draw_configs,draw_boxes,distance_to_bbox_bottom_line,calculate_overlap,filter_model_detector_output,process_video_afterwards_for_debug


if __name__ == '__main__':
    
    base_path = '/home/diego/mydrive/results/7/24/4/lippi_talca_entrada_TESTING5/'
    video_path = f"{base_path}lippi_talca_entrada_TESTING.mkv"
    csv_box_name = f"{base_path}lippi_talca_entrada_TESTING_bbox.csv"
    
    entrance_line=entrance_line = [(643,1050), (341,672)]
    
    process_video_afterwards_for_debug(video_path, csv_box_name, entrance_line_in_out=entrance_line, view_img=True, wait_for_key=False)
    