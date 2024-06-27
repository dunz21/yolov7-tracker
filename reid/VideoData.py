import os
import numpy as np
class VideoData:
    
    def __init__(self):
        self.frame_rate_video = 15
        self.without_video_compression = True
        # self.filter_area = [[1154,353],[1232,353],[1230,563],[1120, 564]] # Parametrized area for filtering detections
        
        
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
        # self.video_time = video_data['video_time']
        
        
        ####### DB ########
        # self.db_host = video_data['db_host']
        # self.db_user = video_data['db_user']
        # self.db_password = video_data['db_password']
        # self.db_name = video_data['db_name']
        
    
    def setBaseFolder(self, base_folder):
        self.base_folder = base_folder
        
    def setClientStoreChannel(self, client_id, store_id, camera_channel_id):
        self.client_id = client_id
        self.store_id = store_id
        self.camera_channel_id = camera_channel_id
        
    def setZoneFilterArea(self, _filter_area):
        if _filter_area is None:
            self.filter_area = None
        else:
            self.filter_area = np.array(_filter_area, np.int32)
        
    def setZoneInOutArea(self, zoneInOutArea):
        self.polygons_in = np.array(zoneInOutArea[0], np.int32)
        self.polygons_out = np.array(zoneInOutArea[1], np.int32)
        self.polygon_area = np.array(zoneInOutArea[2], np.int32)
        
    def setVideoSource(self, video_file_name):
        if self.client_id is None:
            raise Exception("Client ID is not set")
        if self.store_id is None:
            raise Exception("Store ID is not set")
        if self.camera_channel_id is None:
            raise Exception("Channel ID is not set")
        self.source = os.path.join(self.base_folder,str(self.client_id),str(self.store_id),str(self.camera_channel_id), video_file_name)
    
    
    def setVideoMetaInfo(self,video_name, video_date, video_time):
        '''
        video_name: `str`
        video_date: `str` (format: 'YYYY-MM-DD')
        video_time: `str` (format: 'HH:MM:SS')
        '''
        self.name = video_name
        self.video_date = video_date
        self.video_time = video_time
    

    def setDebugVideoSourceCompletePath(self, video_file_name):
        self.source = video_file_name
        