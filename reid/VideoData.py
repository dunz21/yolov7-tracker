import os
import numpy as np
class VideoData:
    
    def __init__(self):
        self.frame_rate_video = 15
        self.without_video_compression = True
        self.filter_area = []
        
        
        # self.source = video_data['source']
        
        ####### DB ########
        # self.polygons_in = video_data['polygons_in']
        # self.polygons_out = video_data['polygons_out']
        # self.polygon_area = video_data['polygon_area']
        
        # self.filter_area = video_data['filter_area']
        
        ####### DB ########
        # self.client_id = video_data['client_id']
        # self.store_id = video_data['store_id']
        
        ####### VIDEO META INFO ########
        # self.name = video_data['name']
        # self.video_date = video_data['video_date']
        # self.start_time_video = video_data['start_time_video']
        
        
        ####### DB ########
        # self.db_host = video_data['db_host']
        # self.db_user = video_data['db_user']
        # self.db_password = video_data['db_password']
        # self.db_name = video_data['db_name']
        
    
    def setBaseFolder(self, base_folder):
        self.base_folder = base_folder
        
    def setClientStoreChannel(self, client_id, store_id, channel_id):
        self.client_id = client_id
        self.store_id = store_id
        self.channel_id = channel_id
        
    def setPolygonArea(self, _in, _out, _area):
        self.polygons_in = np.array(_in, np.int32)
        self.polygons_out = np.array(_out, np.int32)
        self.polygon_area = np.array(_area, np.int32)
        
    def setVideoSource(self, video_file_name):
        if self.client_id is None:
            raise Exception("Client ID is not set")
        if self.store_id is None:
            raise Exception("Store ID is not set")
        if self.channel_id is None:
            raise Exception("Channel ID is not set")
        self.source = os.path.join(self.base_folder,str(self.client_id),str(self.store_id),str(self.channel_id), video_file_name)
    
    def setVideoMetaInfo(self,video_name, video_date, video_time):
        self.name = video_name
        self.video_date = video_date
        self.start_time_video = video_time
        
    def setDB(self, host, user, password, name):
        self.db_host = host
        self.db_user = user
        self.db_password = password
        self.db_name = name

        
        