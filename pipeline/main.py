import os
from pipeline.convert_csv_to_sqlite import convert_csv_to_sqlite
from pipeline.switch_id_fixer import switch_id_corrector_pipeline
from pipeline.image_selection import prepare_data_img_selection, predict_img_selection, clean_img_folder_top_k
from pipeline.vit_pipeline import get_features_from_model
from pipeline.re_ranking import complete_re_ranking
from pipeline.etl_process_visits_per_time import extract_visits_per_hour,save_visits_to_api
from pipeline.etl_process_short_visits_clips import extract_short_visits,process_clips_to_s3,save_short_visits_to_api
from utils.debug_yolo import debug_results_yolo
import logging

def process_complete_pipeline(csv_box_name='', video_path='', img_folder_name='',client_id='',store_id='',video_date='',start_time_video='',frame_rate=''):
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
    logger.info("Step 2: Correct IDs in the database using the switch ID corrector pipeline")
    switch_id_corrector_pipeline(db_path=db_base_path, base_folder_path=img_folder_name, weights='model_weights.pth', model_name='solider')
    logger.info(f"Step 2 completed: Corrected IDs using switch ID corrector pipeline")
    
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
    features = get_features_from_model(model_name='solider', folder_path=f"{img_folder_name}_top4", weights='model_weights.pth', db_path=db_base_path)
    logger.info(f"Step 6 completed: Extracted features from model")
    
    # Step 7: Complete re-ranking using the extracted features
    logger.info("Step 7: Complete re-ranking using the extracted features")
    complete_re_ranking(features, n_images=8, max_number_back_to_compare=57, K1=8, K2=3, LAMBDA=0, db_path=db_base_path)
    logger.info(f"Step 7 completed: Completed re-ranking")
    
    # # Step 8: Extract visits per hour
    # logger.info("Step 8: Extract visits per hour")
    # visits_per_hour = extract_visits_per_hour(db_path=db_base_path, start_time=start_time_video, frame_rate=frame_rate)
    # logger.info(f"Step 8 completed: Extracted visits per hour")
    
    # # Step 9: Save visits per hour to MySQL
    # logger.info("Step 9: Save visits per hour to MySQL")
    # save_visits_to_api(list_visits_group_by_hour=visits_per_hour, store_id=store_id, date=video_date)
    # logger.info(f"Step 9 completed: Saved visits per hour to MySQL")
    
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
    
    # logger.info("Process pipeline completed successfully")


def process_pipeline_mini(csv_box_name='', img_folder_name=''):
    logger = logging.getLogger(__name__)
    
    # Step 1: Convert CSV to SQLite
    logger.info("Step 1: Convert CSV to SQLite")
    db_base_path = csv_box_name.replace('.csv', '.db')
    convert_csv_to_sqlite(csv_file_path=csv_box_name, db_file_path=db_base_path, table_name='bbox_raw')
    logger.info(f"Step 1 completed: Converted {csv_box_name}.csv to SQLite database at {db_base_path}")
    
    # Step 2: Correct IDs in the database using the switch ID corrector pipeline
    logger.info("Step 2: Correct IDs in the database using the switch ID corrector pipeline")
    switch_id_corrector_pipeline(db_path=db_base_path, base_folder_path=img_folder_name, weights='model_weights.pth', model_name='solider')
    logger.info(f"Step 2 completed: Corrected IDs using switch ID corrector pipeline")
    
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
    features = get_features_from_model(model_name='solider', folder_path=f"{img_folder_name}_top4", weights='model_weights.pth', db_path=db_base_path)
    logger.info(f"Step 6 completed: Extracted features from model")
    
    # Step 7: Complete re-ranking using the extracted features
    logger.info("Step 7: Complete re-ranking using the extracted features")
    complete_re_ranking(features, n_images=8, max_number_back_to_compare=57, K1=8, K2=3, LAMBDA=0, db_path=db_base_path)
    logger.info(f"Step 7 completed: Completed re-ranking")
    
    category_summary, unique_id_counts = debug_results_yolo(csv_path=csv_box_name)
    logger.info(f"Category summary: {category_summary}")
    logger.info(f"Unique ID counts: {unique_id_counts}")



