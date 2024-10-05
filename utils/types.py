from enum import Enum

class Direction(Enum):
    In = 'In'
    Out = 'Out'
    Cross = 'Cross'
    Undefined = 'Undefined'

class QueueVideoStatus(Enum):
    DOWNLOADING = 'downloading'
    YOLO = 'yolo'
    REID_FEATURES = 'reid_features'
    VIDEO_ENCODE = 'video_encode'
    UPLOADING_VIDEO_ENCODE = 'uploading_video_encode'
    PENDING = 'pending'
    FAILED = 'failed'
    NOTFOUND = 'not_found'
    FINISHED = 'finished'
