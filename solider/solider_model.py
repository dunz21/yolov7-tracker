import os
from .config import cfg
from .model import make_model

def solider_model(weight_path,device):
    current_file_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    config_path = os.path.join(current_file_path, "configs", "market", "swin_base.yml")  # Construct the relative path to the config
    
    cfg.merge_from_file(config_path)
    cfg.MODEL.SEMANTIC_WEIGHT =  0.2
    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(weight_path)
    model.eval()
    model.to(device)
    return model