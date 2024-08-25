import os
from pipeline.convert_csv_to_sqlite import convert_csv_to_sqlite
from pipeline.switch_id_fixer import switch_id_corrector_pipeline
from pipeline.image_selection import prepare_data_img_selection, predict_img_selection, clean_img_folder_top_k
from pipeline.vit_pipeline import get_features_from_model
from pipeline.re_ranking import complete_re_ranking
from pipeline.etl_process_visits_per_time import extract_visits_per_hour,save_visits_to_api
from pipeline.etl_process_short_visits_clips import extract_short_visits,process_clips_to_s3,save_short_visits_to_api
from pipeline.video_viewer_post_yolo import prepare_event_timestamps_data,save_event_timestamps_to_api
from pipeline.sankey_post_yolo import save_or_update_sankey
from utils.debug_yolo import debug_results_yolo
import logging
from datetime import datetime
from config.api import APIConfig


def process_complete_pipeline(csv_box_name='', video_path='', img_folder_name='',client_id='',store_id='',video_date='',start_time_video='',frame_rate='',solider_weights='model_weights.pth', zone_type_id=1):
    logger = logging.getLogger(__name__)
    
    
    pre_url = 'https://d12y8bglvlc9ab.cloudfront.net'
    # client_id = 1
    # store_id = 3
    # video_date = "2021-06-01"
    # start_time_video = "10:00:00"
    # frame_rate = 15
    bucket_name='videos-mivo'
    
    # connection_mysql = get_connection(HOST, ADMIN, PASS, DB)
    # logger.info(f"Connected to MySQL database at {HOST}")
    
    # Step 1: Convert CSV to SQLite
    logger.info("Step 1: Convert CSV to SQLite")
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
    
    if zone_type_id == 1:
        # Step 6: Extract features from model
        logger.info("Step 6: Extract features from model")
        features = get_features_from_model(model_name='solider', folder_path=f"{img_folder_name}_top4", weights=solider_weights, db_path=db_base_path)
        logger.info(f"Step 6 completed: Extracted features from model")
        
        # Step 7: Complete re-ranking using the extracted features
        logger.info("Step 7: Complete re-ranking using the extracted features")
        complete_re_ranking(features, n_images=8, max_number_back_to_compare=57, K1=8, K2=3, LAMBDA=0, db_path=db_base_path)
        logger.info(f"Step 7 completed: Completed re-ranking")
    
        # Step 8: Extract visits per hour
        logger.info("Step 8: Extract visits per hour")
        visits_per_hour = extract_visits_per_hour(db_path=db_base_path, start_time=start_time_video, frame_rate=frame_rate)
        logger.info(f"Step 8 completed: Extracted visits per hour")
        
        # Step 9: Save visits per hour to MySQL
        logger.info("Step 9: Save visits per hour to MySQL")
        save_visits_to_api(list_visits_group_by_hour=visits_per_hour, store_id=store_id, date=video_date)
        logger.info(f"Step 9 completed: Saved visits per hour to MySQL")

        # Step 10: Extract short visits
        logger.info("Step 10.1: Extract short visits")
        short_visits_clips = extract_short_visits(video_path=video_path, db_path=db_base_path)
        logger.info(f"Step 10.1 completed: Extracted short visits")
        
        # Step 11: Process clips to S3
        logger.info("Step 10.2: Process clips to S3")
        clips_urls = process_clips_to_s3(short_video_clips=short_visits_clips, client_id=client_id, store_id=store_id, date=video_date, pre_url=pre_url, bucket_name=bucket_name)
        logger.info(f"Step 10.2 completed: Processed clips to S3")
        
        # Step 12: Save short visits to MySQL
        logger.info("Step 10.3: Save short visits to MySQL")
        save_short_visits_to_api(short_video_clips_urls=clips_urls, date=video_date, store_id=store_id)
        logger.info(f"Step 10.3 completed: Saved short visits to MySQL")
        
        
        logger.info("Step 10.4: Prepare event timestamps data")
        data = prepare_event_timestamps_data(db_base_path, video_date, start_time_video, store_id)
        save_event_timestamps_to_api(data)
        logger.info(f"Step 10.4 completed: Saved event timestamps to MySQL")
        
        
        
    logger.info("Step 10.5 Prepare event timestamps data")
    save_or_update_sankey(db_path=db_base_path, store_id=store_id, date=video_date, zone_type_id=zone_type_id)
    logger.info(f"Step 10.5 ompleted: Saved event timestamps to MySQL")
    
    
    
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

        # Step 10: Extract short visits
        logger.info("Step 10.1: Extract short visits")
        short_visits_clips = extract_short_visits(video_path=video_path, db_path=db_base_path)
        logger.info(f"Step 10.1 completed: Extracted short visits")
        
        # Step 11: Process clips to S3
        logger.info("Step 10.2: Process clips to S3")
        clips_urls = process_clips_to_s3(short_video_clips=short_visits_clips, client_id=client_id, store_id=store_id, date=video_date, pre_url=pre_url, bucket_name=bucket_name)
        logger.info(f"Step 10.2 completed: Processed clips to S3")
        
        # Step 12: Save short visits to MySQL
        logger.info("Step 10.3: Save short visits to MySQL")
        save_short_visits_to_api(short_video_clips_urls=clips_urls, date=video_date, store_id=store_id)
        logger.info(f"Step 10.3 completed: Saved short visits to MySQL")
        
        
        logger.info("Step 10.4: Prepare event timestamps data")
        data = prepare_event_timestamps_data(db_base_path, video_date, start_time_video,store_id)
        save_event_timestamps_to_api(data)
        logger.info(f"Step 10.4 completed: Saved event timestamps to MySQL")
    
    logger.info("Step 10.5 Prepare sankey data")
    save_or_update_sankey(db_path=db_base_path, store_id=store_id, date=video_date, zone_type_id=zone_type_id)
    logger.info(f"Step 10.5 completed: Saved sankey to MySQL")

    
    logger.info("Process pipeline completed successfully")


def process_pipeline_mini(csv_box_name='', img_folder_name='',solider_weights='model_weights.pth',override_db_name=None):
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
    
    # Step 6: Extract features from model
    logger.info("Step 6: Extract features from model")
    features = get_features_from_model(model_name='solider', folder_path=f"{img_folder_name}", weights=solider_weights, db_path=db_base_path)
    logger.info(f"Step 6 completed: Extracted features from model")
    
    # Step 7: Complete re-ranking using the extracted features
    logger.info("Step 7: Complete re-ranking using the extracted features")
    complete_re_ranking(features, n_images=8, max_number_back_to_compare=57, K1=8, K2=3, LAMBDA=0, db_path=db_base_path)
    logger.info(f"Step 7 completed: Completed re-ranking")
    
### DIEGO TEST ###


def process_pipeline_by_dates(base_result_path, client_id, store_id, camera_channel_id, start_date, end_date):
    # Build the base path
    base_path = os.path.join(base_result_path, str(client_id), str(store_id), str(camera_channel_id))
    
    # Convert the dates to datetime objects for comparison
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    
    # Check if the base path exists
    if not os.path.exists(base_path):
        print(f"Path {base_path} does not exist.")
        return
    
    # Retrieve and sort the folders by date
    folders = []
    for folder_name in os.listdir(base_path):
        parts = folder_name.split('_')
        try:
            folder_date = datetime.strptime(parts[-2], '%Y%m%d')
            folders.append((folder_name, folder_date))
        except (ValueError, IndexError):
            continue

    # Sort folders by the extracted folder date
    folders = sorted(folders, key=lambda x: x[1])
    
    # Iterate through the sorted folders
    for folder_name, folder_date in folders:
        # Check if the folder date is within the date range
        if start_date <= folder_date <= end_date:
            folder_path = os.path.join(base_path, folder_name)
            start_time_video = folder_name.split('_')[-1]
            # print(f"Processing folder: {folder_name} at {folder_path}")
            print(f"Date: {folder_date.strftime('%Y-%m-%d')}, Start Time Video: {start_time_video}, folder name: {folder_name}, folder path: {folder_path}")
            # Add your processing logic here

            
if __name__ == '__main__':
    # base_url_api = 'https://api-v1.mivo.cl'
    base_url_api = os.getenv('BASE_URL_API', 'http://localhost:1001')  
    APIConfig.initialize(base_url_api)
    base_result_path = os.getenv('RESULTS_ROOT_FOLDER_PATH', '')
    client_id = 1
    store_id = 10
    camera_channel_id = 8
    start_date = '20240818'
    end_date = '20240818'
    
    process_pipeline_by_dates(base_result_path, client_id, store_id, camera_channel_id, start_date, end_date)

    
### DIEGO TEST ###    
    
    
    
    
### SANKEY
# if __name__ == '__main__':
#     base_url_api = os.getenv('BASE_URL_API', 'http://localhost:1001')
#     base_url_api = 'https://api-v1.mivo.cl'
#     APIConfig.initialize(base_url_api)
#     save_or_update_sankey(
#         # db_path='/home/diego/mydrive/results/1/10/8/apumanque_entrada_2_20240731_1000/apumanque_entrada_2_20240731_1000_bbox.db', 
#         db_path='/home/diego/mydrive/results/1/10/2/apumanque_puerta_1_20240730_1000/apumanque_puerta_1_20240730_1000_bbox.db', 
#         store_id=10, 
#         date='2024-07-30',
#         zone_type_id=2
#         )

### PIPIELINE MINI
# if __name__ == '__main__':
#     solider_weights = 'transformer_120.pth'
#     base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240820_1000_CFPS/'
#     csv_file_name = os.path.join(base_result_folder, 'costanera_entrada_20240820_1000_CFPS_bbox.csv')
#     img_folder_name = os.path.join(base_result_folder, 'imgs')
#     features = process_pipeline_mini(
#         csv_box_name=csv_file_name,
#         img_folder_name=img_folder_name,
#         solider_weights=solider_weights,
#         )
#     base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240705_1000_CFPS/'
#     csv_file_name = os.path.join(base_result_folder, 'costanera_entrada_20240705_1000_CFPS_bbox.csv')
#     img_folder_name = os.path.join(base_result_folder, 'imgs')
#     features = process_pipeline_mini(
#         csv_box_name=csv_file_name,
#         img_folder_name=img_folder_name,
#         solider_weights=solider_weights,
#         )
#     base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240706_1000_CFPS/'
#     csv_file_name = os.path.join(base_result_folder, 'costanera_entrada_20240706_1000_CFPS_bbox.csv')
#     img_folder_name = os.path.join(base_result_folder, 'imgs')
#     features = process_pipeline_mini(
#         csv_box_name=csv_file_name,
#         img_folder_name=img_folder_name,
#         solider_weights=solider_weights,
#         )
    
#     base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240701_1000_CFPS/'
#     csv_file_name = os.path.join(base_result_folder, 'costanera_entrada_20240701_1000_CFPS_bbox.csv')
#     img_folder_name = os.path.join(base_result_folder, 'imgs')
#     features = process_pipeline_mini(
#         csv_box_name=csv_file_name,
#         img_folder_name=img_folder_name,
#         solider_weights=solider_weights,
#         )
    
#     base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240702_1000_CFPS/'
#     csv_file_name = os.path.join(base_result_folder, 'costanera_entrada_20240702_1000_CFPS_bbox.csv')
#     img_folder_name = os.path.join(base_result_folder, 'imgs')
#     features = process_pipeline_mini(
#         csv_box_name=csv_file_name,
#         img_folder_name=img_folder_name,
#         solider_weights=solider_weights,
#         )
    

    
    
### CORRER PARA VISITAS CORTAS 
# if __name__ == '__main__':
#     base_url_api = os.getenv('BASE_URL_API', 'http://localhost:1001')
#     SOLIDER_WEIGHTS ='model_weights.pth'
#     APIConfig.initialize(base_url_api)
#     solider_weights = '/home/diego/Documents/solider-reid/log/mivo/test2_complete/transformer_120.pth'
#     # base_result_folder = '/home/diego/Documents/MivoRepos/mivo-project/apumanque-results/apumanque_entrada_2_20240720_1000'
#     base_result_folder = '/home/diego/Documents/MivoRepos/mivo-project/apumanque-results/apumanque_entrada_2_20240723_1000/'
#     csv_file_name = os.path.join(base_result_folder, 'apumanque_entrada_2_20240723_1000_bbox.csv')
#     img_folder_name = os.path.join(base_result_folder, 'imgs')
#     features = process_complete_pipeline(
#         csv_box_name=csv_file_name,
#         video_path='/home/diego/Documents/MivoRepos/mivo-project/apumanque-footage/apumanque_entrada_2_20240723_1000.mkv',
#         img_folder_name=img_folder_name,
#         solider_weights=solider_weights,
#         client_id=1,
#         store_id=10,
#         video_date='2024-07-23',
#         start_time_video='09:00:00',
#         frame_rate=15
#         )
    
    
    
# SAVE TO BD VISITS
# if __name__ == '__main__':
#     # base_url_api = os.getenv('BASE_URL_API', 'http://localhost:1001')
#     base_url_api = 'https://api-v1.mivo.cl/'
#     APIConfig.initialize(base_url_api)
#     client_id = 3
#     store_id = 16
#     frame_rate = 15

#     zone_type_id=1
    
    
    ##### ALL THE FOLDER ####
    # base_path = f'/home/diego/mydrive/results/{client_id}/{store_id}/8/'
    # footage_path = f'/home/diego/mydrive/footage/{client_id}/{store_id}/8'
    # for folder_name in os.listdir(base_path):
    #     folder_path = os.path.join(base_path, folder_name)
    #     if os.path.isdir(folder_path) and folder_name not in ['OTROS', 'OLD']:
    #         parts = folder_name.split('_')
    #         date_str = parts[3]  # The date part
    #         time_str = parts[4]  # The time part
    #         video_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
    #         start_time_video = datetime.strptime(time_str, '%H%M').strftime('%H:%M:%S')
    #         db_base_path = os.path.join(base_path, folder_name, f'{folder_name}_bbox.db')
    #         video_path = os.path.join(footage_path, f'{folder_name}.mkv')
            
    #         print(f"\n Processing {folder_name}")
    #         print(f"DB path: {db_base_path}")
    #         print(f"Video path: {video_path}")
    #         print(f"Video date: {video_date}")
    #         print(f"Start time video: {start_time_video}")
    #         data = prepare_event_timestamps_data(db_base_path, video_date, start_time_video,store_id)
    #         save_event_timestamps_to_api(data)
    
    
    
    
    # base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240820_1000_CFPS/'
    # db_base_path = os.path.join(base_result_folder, 'costanera_entrada_20240820_1000_CFPS_bbox.db')
    # video_path = '/home/diego/mydrive/footage/3/16/3/costanera_entrada_20240820_1000_CFPS.mkv'
    # video_date = '2024-08-20'
    # start_time_video = '10:00:00'
    # features = process_save_bd_pipeline(db_base_path=db_base_path, video_path=video_path, client_id=client_id, store_id=store_id, video_date=video_date, start_time_video=start_time_video, frame_rate=frame_rate,zone_type_id=zone_type_id)
    # base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240724_1000_CFPS'
    # db_base_path = os.path.join(base_result_folder, 'costanera_entrada_20240724_1000_CFPS_bbox.db')
    # video_path = '/home/diego/mydrive/footage/3/16/3/costanera_entrada_20240724_1000_CFPS.mkv'
    # video_date = '2024-07-24'
    # start_time_video = '10:00:00'
    # features = process_save_bd_pipeline(db_base_path=db_base_path, video_path=video_path, client_id=client_id, store_id=store_id, video_date=video_date, start_time_video=start_time_video, frame_rate=frame_rate,zone_type_id=zone_type_id)
    # base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240725_1012_CFPS'
    # db_base_path = os.path.join(base_result_folder, 'costanera_entrada_20240725_1012_CFPS_bbox.db')
    # video_path = '/home/diego/mydrive/footage/3/16/3/costanera_entrada_20240725_1012_CFPS.mkv'
    # video_date = '2024-07-25'
    # start_time_video = '10:00:00'
    # features = process_save_bd_pipeline(db_base_path=db_base_path, video_path=video_path, client_id=client_id, store_id=store_id, video_date=video_date, start_time_video=start_time_video, frame_rate=frame_rate,zone_type_id=zone_type_id)
    # base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240726_1000_CFPS'
    # db_base_path = os.path.join(base_result_folder, 'costanera_entrada_20240726_1000_CFPS_bbox.db')
    # video_path = '/home/diego/mydrive/footage/3/16/3/costanera_entrada_20240726_1000_CFPS.mkv'
    # video_date = '2024-07-26'
    # start_time_video = '10:00:00'
    # features = process_save_bd_pipeline(db_base_path=db_base_path, video_path=video_path, client_id=client_id, store_id=store_id, video_date=video_date, start_time_video=start_time_video, frame_rate=frame_rate,zone_type_id=zone_type_id)
    # base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240727_1000_CFPS'
    # db_base_path = os.path.join(base_result_folder, 'costanera_entrada_20240727_1000_CFPS_bbox.db')
    # video_path = '/home/diego/mydrive/footage/3/16/3/costanera_entrada_20240727_1000_CFPS.mkv'
    # video_date = '2024-07-27'
    # start_time_video = '10:00:00'
    # features = process_save_bd_pipeline(db_base_path=db_base_path, video_path=video_path, client_id=client_id, store_id=store_id, video_date=video_date, start_time_video=start_time_video, frame_rate=frame_rate,zone_type_id=zone_type_id)
    # base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240719_1000_CFPS'
    # db_base_path = os.path.join(base_result_folder, 'costanera_entrada_20240719_1000_CFPS_bbox.db')
    # video_path = '/home/diego/mydrive/footage/3/16/3/costanera_entrada_20240719_1000_CFPS.mkv'
    # video_date = '2024-07-19'
    # start_time_video = '10:00:00'
    # features = process_save_bd_pipeline(db_base_path=db_base_path, video_path=video_path, client_id=client_id, store_id=store_id, video_date=video_date, start_time_video=start_time_video, frame_rate=frame_rate,zone_type_id=zone_type_id)
    # base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240720_1000_CFPS'
    # db_base_path = os.path.join(base_result_folder, 'costanera_entrada_20240720_1000_CFPS_bbox.db')
    # video_path = '/home/diego/mydrive/footage/3/16/3/costanera_entrada_20240720_1000_CFPS.mkv'
    # video_date = '2024-07-20'
    # start_time_video = '10:00:00'
    # features = process_save_bd_pipeline(db_base_path=db_base_path, video_path=video_path, client_id=client_id, store_id=store_id, video_date=video_date, start_time_video=start_time_video, frame_rate=frame_rate,zone_type_id=zone_type_id)
    # base_result_folder = '/home/diego/mydrive/results/3/16/3/costanera_entrada_20240721_1000_CFPS'
    # db_base_path = os.path.join(base_result_folder, 'costanera_entrada_20240721_1000_CFPS_bbox.db')
    # video_path = '/home/diego/mydrive/footage/3/16/3/costanera_entrada_20240721_1000_CFPS.mkv'
    # video_date = '2024-07-21'
    # start_time_video = '10:00:00'
    # features = process_save_bd_pipeline(db_base_path=db_base_path, video_path=video_path, client_id=client_id, store_id=store_id, video_date=video_date, start_time_video=start_time_video, frame_rate=frame_rate,zone_type_id=zone_type_id)