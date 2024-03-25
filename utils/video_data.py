import numpy as np
def get_video_data():
    data= [
        {
            'name' : "conce",
            'source' : "/home/diego/Documents/Footage/CONCEPCION_CH1.mp4",
            'description' : "Video de Conce",
            'folder_img' : "imgs_conce",
            'polygons_in' : np.array([[265, 866],[583, 637],[671, 686],[344, 948]], np.int32),
            'polygons_out' : np.array([[202, 794],[508, 608],[575, 646],[263, 865]], np.int32),
            'polygon_area' : np.array([[0,1080],[0,600],[510,500],[593,523],[603,635],[632,653],[738,588],[756,860],[587,1080]], np.int32),
        },
        {
            'name' : "santos_dumont",
            'source' : "/home/diego/Documents/Footage/SANTOS LAN_ch6.mp4",
            'description' : "Video de Santos Dumont",
            'folder_img' : "imgs_santos_dumont",
            'polygons_in' : np.array([[865, 510],[1117,550],[1115,595],[831,541]], np.int32),
            'polygons_out' : np.array([[894, 480],[1118,510],[1117,550],[865,510]], np.int32),
            'polygon_area' : np.array([[710,511],[712,650],[1119,757],[1206,562],[1179,378],[731,325]], np.int32),
        },
        {
            'name' : "santos_dumont_debug",
            'source' : "/home/diego/Documents/Footage/reid_qa/dumont_debug_tracker_2_335.mp4",
            'description' : "Video de Santos Dumont",
            'folder_img' : "imgs_santos_dumont_debug",
            'polygons_in' : np.array([[865, 510],[1117,550],[1115,595],[831,541]], np.int32),
            'polygons_out' : np.array([[894, 480],[1118,510],[1117,550],[865,510]], np.int32),
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
            'source' : "/home/diego/Documents/Footage/conce_logic_in_out_4.mp4",
            'description' : "Video de Conce",
            'folder_img' : "imgs_conce_debug",
            'polygons_in' : np.array([[265, 866],[583, 637],[671, 686],[344, 948]], np.int32),
            'polygons_out' : np.array([[202, 794],[508, 608],[575, 646],[263, 865]], np.int32),
            'polygon_area' : np.array([[0,1080],[0,600],[510,500],[593,523],[603,635],[632,653],[738,588],[756,860],[587,1080]], np.int32),
        },
    ]
    return data    

