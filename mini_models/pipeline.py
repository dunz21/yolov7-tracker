import os
from tqdm import tqdm
from utils.pipeline import getFinalScore,get_features_from_model
from mini_models.re_ranking import complete_re_ranking, generate_re_ranking_html_report,classification_match
from utils.tools import convert_csv_to_sqlite,prepare_data_img_selection,predict_img_selection,clean_img_folder_top_k


if __name__ == '__main__':
    MODEL_WEIGHT = 'mini_models/results/image_selection_model.pkl'
    SOLIDER_MODEL_PATH = '/home/diego/Documents/tracklab/yolov7-tracker/model_weights.pth'
    ROOT_FOLDER = "runs/detect/"
    
    #ACA SE CAMBIA
    RUN_FOLDER = '2024_04_03_conce_debug_switch_id' 
    base_folder_images = 'imgs_conce_debug'
    dest_folder_results = 'imgs_conce_debug_top4'
    CSV_FILE = 'conce_debug_bbox.db'
    FEAT_FILE = 'conce_debug_bbox.csv'
    #ACA SE CAMBIA
    
    ROOT_FOLDER = os.path.join(ROOT_FOLDER, RUN_FOLDER)
    base_folder_images = os.path.join(ROOT_FOLDER, base_folder_images)
    dest_folder_results = os.path.join(ROOT_FOLDER, dest_folder_results)
    DB_FILE_PATH = os.path.join(ROOT_FOLDER, CSV_FILE)
    
    
    DISTANCE_METHOD = "cosine"
    features_file = os.path.join(ROOT_FOLDER, FEAT_FILE)
    
    k_fold = 4
    threshold = 0.9
    
    
    FRAME_RATE = 15
    n_images = 8
    max_number_back_to_compare = 57
    K1 = 8
    K2 = 3
    LAMBDA = 0
    save_csv_dir = ROOT_FOLDER

    
    
    with tqdm(total=8, desc="Overall Progress", unit="step") as pbar:
        
        # 1.- Prepare data for image selection
        # prepare_data_img_selection(db_path=DB_FILE_PATH, origin_table="bbox_raw", k_folds=4, n_images=5, new_table_name="bbox_img_selection")
        # pbar.update(1)

        # # 2.- Predict which images are good
        # bbox_img_selection = predict_img_selection(db_file_path=DB_FILE_PATH, model_weights_path=MODEL_WEIGHT)
        # pbar.update(1)

        # # 3.- Apply image separation based on model results
        # clean_img_folder_top_k(db_file_path=DB_FILE_PATH, base_folder_images=base_folder_images, dest_folder_results=dest_folder_results, k_fold=4, threshold=0.9)
        # pbar.update(1)
        
        # 4.- Get features from the model
        features = get_features_from_model(folder_path=dest_folder_results, weights=SOLIDER_MODEL_PATH, model_name='solider', db_path=DB_FILE_PATH)
        pbar.update(1) 
        exit(0)
        results, file_name, posible_pair_matches = complete_re_ranking(features,
                                        n_images=n_images,
                                        max_number_back_to_compare=max_number_back_to_compare,
                                        K1=K1,
                                        K2=K2,
                                        LAMBDA=LAMBDA,
                                        filter_known_matches=None,
                                        save_csv_dir=save_csv_dir)
        pbar.update(1) 
        
        RE_RANK_HTML = os.path.join(save_csv_dir, f'{file_name}.html')
        generate_re_ranking_html_report(results, base_folder_images, FRAME_RATE, RE_RANK_HTML)
        pbar.update(1)         
        
        final_classifier = classification_match(posible_pair_matches=posible_pair_matches,filename_csv=f"{ROOT_FOLDER}/auto_match.csv",db_path=f"{ROOT_FOLDER}/santos_dumont_bbox.db")
        print(final_classifier)
        

