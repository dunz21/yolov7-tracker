import os
import re
from pipeline.convert_csv_to_sqlite import convert_csv_to_sqlite
from pipeline.switch_id_fixer import switch_id_corrector_pipeline
from pipeline.image_selection import prepare_data_img_selection, predict_img_selection, clean_img_folder_top_k
from pipeline.vit_pipeline import get_features_from_model
from pipeline.re_ranking import complete_re_ranking
from pipeline.etl_process_visits_per_time import extract_visits_per_hour,save_visits_to_api
from pipeline.etl_process_short_visits_clips import extract_short_visits,process_clips_to_s3,save_short_visits_to_api
from pipeline.event_timestamp_post_yolo import save_event_timestamps
from pipeline.sankey_post_yolo import save_or_update_sankey
from pipeline.reid_matches_post_yolo import save_or_update_reid_matches
from utils.debug_yolo import debug_results_yolo
import logging
from datetime import datetime
from config.api import APIConfig


def process_complete_pipeline(csv_box_name='', video_path='', img_folder_name='',client_id='',store_id='',video_date='',start_time_video='',frame_rate='',solider_weights='model_weights.pth', zone_type_id=1):
    logger = logging.getLogger(__name__)
    db_base_path = process_pipeline_mini(csv_box_name=csv_box_name, img_folder_name=img_folder_name,solider_weights=solider_weights,zone_type_id=zone_type_id)
    process_save_bd_pipeline(db_base_path=db_base_path, video_path=video_path, client_id=client_id,store_id=store_id,video_date=video_date,start_time_video=start_time_video,frame_rate=frame_rate,zone_type_id=zone_type_id)
    logger.info("Process pipeline completed successfully")
    
    
def process_save_bd_pipeline(db_base_path='', video_path='', client_id='',store_id='',video_date='',start_time_video='',frame_rate='',zone_type_id=1):
    logger = logging.getLogger(__name__)
    
    pre_url = 'https://d12y8bglvlc9ab.cloudfront.net'
    bucket_name='videos-mivo'

    if zone_type_id==1:
        # Step 8: Extract visits per hour
        logger.info("Step 8: Extract visits per hour")
        visits_per_hour = extract_visits_per_hour(db_path=db_base_path, start_time=start_time_video, frame_rate=frame_rate)
        logger.info(f"Step 8 completed: Extracted visits per hour")
        
        # Step 9: Save visits per hour to MySQL
        logger.info("Step 9: Save visits per hour to MySQL")
        save_visits_to_api(list_visits_group_by_hour=visits_per_hour, store_id=store_id, date=video_date)
        logger.info(f"Step 9 completed: Saved visits per hour to MySQL")

        # # Step 10: Extract short visits
        # logger.info("Step 10.1: Extract short visits")
        # short_visits_clips = extract_short_visits(video_path=video_path, db_path=db_base_path)
        # logger.info(f"Step 10.1 completed: Extracted short visits")
        
        # # Step 11: Process clips to S3
        # logger.info("Step 10.2: Process clips to S3")
        # clips_urls = process_clips_to_s3(short_video_clips=short_visits_clips, client_id=client_id, store_id=store_id, date=video_date, pre_url=pre_url, bucket_name=bucket_name)
        # logger.info(f"Step 10.2 completed: Processed clips to S3")
        
        # # Step 12: Save short visits to MySQL
        # logger.info("Step 10.3: Save short visits to MySQL")
        # save_short_visits_to_api(short_video_clips_urls=clips_urls, date=video_date, store_id=store_id)
        # logger.info(f"Step 10.3 completed: Saved short visits to MySQL")
        
        
        logger.info("Step 10.4: Prepare event timestamps data")
        save_event_timestamps(db_path=db_base_path, date=video_date, start_video_time=start_time_video, store_id=store_id)
        logger.info(f"Step 10.4 completed: Saved event timestamps to MySQL")
        
        logger.info("Step 10.5: Prepare reid matches data")
        save_or_update_reid_matches(db_path=db_base_path,store_id=store_id, date=video_date)
        logger.info(f"Step 10.5 completed: Saved reid matches to MySQL")
    
    logger.info("Step 10.6 Prepare event timestamps data")
    save_or_update_sankey(db_path=db_base_path, store_id=store_id, date=video_date, zone_type_id=zone_type_id)
    logger.info(f"Step 10.6 ompleted: Saved event timestamps to MySQL")

    
    logger.info("Process pipeline completed successfully")


def process_pipeline_mini(csv_box_name='', img_folder_name='',solider_weights='model_weights.pth',override_db_name=None, zone_type_id=1):
    logger = logging.getLogger(__name__)
    
    category_summary, unique_id_counts = debug_results_yolo(csv_path=csv_box_name)
    logger.info(f"Category summary: {category_summary}")
    logger.info(f"Unique ID counts: {unique_id_counts}")
    
    # Step 1: Convert CSV to SQLite
    logger.info("Step 1: Convert CSV to SQLite")
    if override_db_name:
        db_base_path = override_db_name
    else:
        db_base_path = csv_box_name.replace('.csv', '.db')
    convert_csv_to_sqlite(csv_file_path=csv_box_name, db_file_path=db_base_path, table_name='bbox_raw')
    logger.info(f"Step 1 completed: Converted {csv_box_name}.csv to SQLite database at {db_base_path}")
    
    # Step 2: Correct IDs in the database using the switch ID corrector pipeline
    # logger.info("Step 2: Correct IDs in the database using the switch ID corrector pipeline")
    # switch_id_corrector_pipeline(db_path=db_base_path, base_folder_path=img_folder_name, weights=solider_weights, model_name='solider')
    # logger.info(f"Step 2 completed: Corrected IDs using switch ID corrector pipeline")
    
    # Step 3: Prepare data for image selection
    logger.info("Step 3: Prepare data for image selection")
    prepare_data_img_selection(db_path=db_base_path, origin_table="bbox_raw", k_folds=4, n_images=5, new_table_name="bbox_img_selection")
    logger.info(f"Step 3 completed: Prepared data for image selection")
    
    # Step 4: Predict image selection
    logger.info("Step 4: Predict image selection")
    predict_img_selection(db_file_path=db_base_path, model_weights_path='mini_models/results/image_selection_model.pkl')
    logger.info(f"Step 4 completed: Predicted image selection")
    
    # Step 5: Clean image folder by selecting top-k images
    logger.info("Step 5: Clean image folder by selecting top-k images")
    clean_img_folder_top_k(db_file_path=db_base_path, base_folder_images=img_folder_name, dest_folder_results=f"{img_folder_name}_top4", k_fold=4, threshold=0.9)
    logger.info(f"Step 5 completed: Cleaned image folder and selected top-k images")
    
    if zone_type_id == 1: # Only for entrance, prevent to run for exterior
        # Step 6: Extract features from model
        logger.info("Step 6: Extract features from model")
        features = get_features_from_model(model_name='solider', folder_path=f"{img_folder_name}", weights=solider_weights, db_path=db_base_path)
        logger.info(f"Step 6 completed: Extracted features from model")
        
        # Step 7: Complete re-ranking using the extracted features
        logger.info("Step 7: Complete re-ranking using the extracted features")
        complete_re_ranking(features, n_images=8, max_number_back_to_compare=57, K1=8, K2=3, LAMBDA=0, db_path=db_base_path)
        logger.info(f"Step 7 completed: Completed re-ranking")
        
    return db_base_path
    
### DIEGO TEST ###


def process_pipeline_by_dates(base_result_path='',base_footage_path='', client_id='', store_id='', camera_channel_id='', start_date='', end_date='', zone_type_id=1, processes_to_execute=[]):
    # Build the base path
    base_path_results = os.path.join(base_result_path, str(client_id), str(store_id), str(camera_channel_id))
    base_footage_raw_path = os.path.join(base_footage_path, str(client_id), str(store_id), str(camera_channel_id))
    
    # Convert the dates to datetime objects for comparison
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    
    # Check if the base path exists
    if not os.path.exists(base_path_results):
        print(f"Path {base_path_results} does not exist.")
        return
    
    # Retrieve and store the folders that match the pattern
    pattern = re.compile(r'(\d{8})_(\d{4})')
    folders = []
    for folder_name in os.listdir(base_path_results):
        match = pattern.search(folder_name)
        if match:
            try:
                folder_date_str = match.group(1)  # Extract the date part (first group)
                folder_time_str = match.group(2)  # Extract the time part (second group)
                
                # Convert the folder_date_str to a datetime object
                folder_date = datetime.strptime(folder_date_str, '%Y%m%d')
                folder_time_str = datetime.strptime(folder_time_str, '%H%M').strftime('%H:%M:%S')
                
                # Append the folder name, date, and time to the list
                folders.append((folder_name, folder_date, folder_time_str))
            except (ValueError, IndexError):
                continue

    # Sort folders by the extracted folder date
    folders = sorted(folders, key=lambda x: x[1])
    
    # Iterate through the sorted folders
    for folder_name, folder_date,start_time_video in folders:
        # Check if the folder date is within the date range
        if start_date <= folder_date <= end_date:
            folder_path = os.path.join(base_path_results, folder_name)
            folder_footage_raw_path = os.path.join(base_footage_raw_path, folder_name)
            video_date = folder_date.strftime('%Y-%m-%d')
            csv_box_name = os.path.join(folder_path, f'{folder_name}_bbox.csv')
            img_folder = os.path.join(folder_path, 'imgs')
            video_file = os.path.join(f'{folder_footage_raw_path}.mkv')
            print(f"Processing folder: {folder_name}")
            
            # Add your processing logic here
            db_path = os.path.join(folder_path, f'{folder_name}_bbox.db')
            if not os.path.exists(db_path):
                print(f"DB path {db_path} does not exist for folder {folder_name}. Skipping.")
                continue
            
            
            if 'reid_matches' in processes_to_execute: # tiempo en tienda para el exponential chart
                save_or_update_reid_matches(db_path=db_path, store_id=store_id, date=video_date)
            if 'event_timestamps' in processes_to_execute:
                save_event_timestamps(db_path=db_path, date=video_date, start_video_time=start_time_video, store_id=store_id)
            if 'sankey' in processes_to_execute:
                save_or_update_sankey(db_path=db_path, store_id=store_id, date=video_date, zone_type_id=zone_type_id)
            if 'process_pipeline_mini' in processes_to_execute:
                process_pipeline_mini(csv_box_name=csv_box_name, img_folder_name=img_folder)
            if 'process_save_bd_pipeline' in processes_to_execute:
                process_save_bd_pipeline(db_base_path=db_path, video_path=video_file, client_id=client_id,store_id=store_id,video_date=video_date,start_time_video=start_time_video,frame_rate=15,zone_type_id=zone_type_id)

            
            
if __name__ == '__main__':
    
    PRODUCTION_MODE = True
    
    base_url_api = 'https://api-v1.mivo.cl' if PRODUCTION_MODE else os.getenv('BASE_URL_API', 'http://localhost:1001')
    APIConfig.initialize(base_url_api)
    base_result_path = os.getenv('RESULTS_ROOT_FOLDER_PATH', '')
    base_footage_path = os.getenv('FOOTAGE_ROOT_FOLDER_PATH', '')
    start_date = '20240923'
    end_date = '20240925'
    
    
    # Para sacar la data del sankey, primero procesar KUNA con el channel correcto y eso videos moverlo a otra carpeta de channel_camera_id
    # Para sacar la data del sankey de diponti dejar zone_type_id=3 y apuntar a la camara 2
    
    
    # client_id, store_id, camera_channel_id, zone_type_id, processes_to_execute = 1, 10, 8, 1,['reid_matches','event_timestamps','sankey'] #LDP
    # client_id, store_id, camera_channel_id, zone_type_id, processes_to_execute = 1, 10, 2, 3, ['sankey'] #LDP Exterior
    # client_id, store_id, camera_channel_id, zone_type_id, processes_to_execute = 3, 16, 3, 1,['reid_matches','event_timestamps','sankey'] # KUNA
    # client_id, store_id, camera_channel_id, zone_type_id,processes_to_execute = 3, 16, 999, 1, ['process_pipeline_mini']# KUNA Exterior
    
    
    # client_id, store_id, camera_channel_id, zone_type_id, processes_to_execute = 1, 10, 8, 1,['process_save_bd_pipeline'] #LDP
    
    
    # client_id, store_id, camera_channel_id, zone_type_id, processes_to_execute = 7, 22, 1, 1,['process_pipeline_mini','process_save_bd_pipeline'] # Costanera
    client_id, store_id, camera_channel_id, zone_type_id, processes_to_execute = 7, 24, 4, 1,['process_pipeline_mini','process_save_bd_pipeline'] # Talca
    
    

    
    process_pipeline_by_dates(base_result_path,base_footage_path, client_id, store_id, camera_channel_id, start_date, end_date, zone_type_id, processes_to_execute)

    
