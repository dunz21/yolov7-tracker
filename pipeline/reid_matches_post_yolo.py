import os
from config.api import APIConfig
from reid.matches import extract_reid_matches


def save_or_update_reid_matches(db_path='',store_id=0, date='', reid_matches=[]):
    reid_matches = extract_reid_matches(db_path, max_distance=0.4, min_time_diff='00:00:10', max_time_diff='01:00:00', fps=15)
    APIConfig.save_reid_matches(store_id, date, reid_matches=reid_matches.to_dict(orient='records'))

    


        
