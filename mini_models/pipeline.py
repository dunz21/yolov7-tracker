import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving and loading the model
from sklearn.model_selection import KFold
import shutil
import sqlite3
from utils.pipeline import getFinalScore,get_features_from_model
from mini_models.re_ranking import complete_re_ranking, generate_re_ranking_html_report
from utils.tools import convert_csv_to_sqlite,prepare_data_img_selection


    
def train_and_save_model(training_data_path, model_save_path):
    INTEREST_LABEL = 'label_img'
    interest_values = [1, 2]

    only_train_data = pd.read_csv(training_data_path)
    only_train_data = only_train_data.dropna(subset=['img_name'])
    only_train_data = only_train_data[only_train_data[INTEREST_LABEL].isin(interest_values)]

    features = ['area', 'centroid_x', 'centroid_y', 'frame_number', 'overlap', 'distance_to_center', 'conf_score']
    target = INTEREST_LABEL

    assert all(f in only_train_data.columns for f in features + [target]), "Some required columns are missing."

    X_train, X_val, y_train, y_val = train_test_split(only_train_data[features], only_train_data[target], test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_save_path)

    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy: {val_accuracy}")

def predict_img_selection(db_file_path='', model_weights_path='', export_csv=False, csv_dir=None):
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # Load data from the database
    data = pd.read_sql('SELECT * FROM bbox_img_selection', conn)
    features = ['area', 'centroid_x', 'centroid_y', 'frame_number', 'overlap', 'distance_to_center', 'conf_score']
    
    # Ensure the input data contains all the required features
    assert all(f in data.columns for f in features), "Some required columns for prediction are missing."
    
    # Load the model
    model = joblib.load(model_weights_path)
    
    # Perform predictions
    predictions = model.predict(data[features])
    predicted_confidences = model.predict_proba(data[features]).max(axis=1)
    
    # Update the database
    # TODO: FIX ACA ESTA haciendo update a muchos IDS
    for i, row in data.iterrows():
        sql = """UPDATE bbox_img_selection SET model_label_img = ?, model_label_conf = ? WHERE id = ? and img_name = ?"""
        cursor.execute(sql, (int(predictions[i]), round(predicted_confidences[i], 2), row['id'], row['img_name']))
    
    conn.commit()
    
    if export_csv:
        if csv_dir is None:
            raise ValueError("csv_dir must be specified if export_csv is True")
        # Export to CSV
        data['model_label_img'] = predictions
        data['model_label_conf'] = predicted_confidences.round(2)
        csv_path = os.path.join(csv_dir, "img_selection_predicted.csv")
        data.to_csv(csv_path, index=False)
        print(f"Predictions exported to CSV at: {csv_path}")
    
    conn.close()

def clean_img_folder_top_k(db_file_path='', base_folder_images='', dest_folder_results='', k_fold=5, threshold=0.9, export_csv=False, csv_dir=None):
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # Load the modified data from the database
    data = pd.read_sql('SELECT * FROM bbox_img_selection WHERE img_name IS NOT NULL', conn)
    data.sort_values(by=['id', 'frame_number'], inplace=True)
    
    for id_value in data['id'].unique():
        local_threshold = threshold
        while True:
            id_df = data[data['id'] == id_value]
            filtered_id_df = id_df[(id_df['model_label_conf'] > local_threshold) & (id_df['model_label_img'] == 2)].copy()
            
            if len(filtered_id_df) >= k_fold or local_threshold <= 0:
                break
            local_threshold -= 0.05
        
        if len(filtered_id_df) >= k_fold:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=k_fold)
            
            for fold_number, (_, test_index) in enumerate(kf.split(filtered_id_df), start=1):
                fold_df = filtered_id_df.iloc[test_index]
                selected_row = fold_df.sample(n=1)
                selected_index = selected_row.index[0]
                
                # Update the database with fold and selection information
                sql = """UPDATE bbox_img_selection SET k_fold_selection = ?, selected_image = ? WHERE id = ? and img_name = ?"""
                cursor.execute(sql, (int(fold_number), True, int(selected_row.iloc[0]['id']),selected_row.iloc[0]['img_name']))
                
                # # Copy the selected image
                id_img = selected_row.iloc[0]['img_name'].split('_')[1]
                source_path = os.path.join(base_folder_images, id_img ,selected_row.iloc[0]['img_name'])
                dest_path = source_path.replace(base_folder_images, dest_folder_results)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(source_path, dest_path)
    
    conn.commit()
    
    if export_csv:
        if csv_dir is None:
            raise ValueError("csv_dir must be specified if export_csv is True")
        # Export to CSV
        csv_path = os.path.join(csv_dir, "cleaned_images_selection.csv")
        data.to_csv(csv_path, index=False)
        print(f"Cleaned data exported to CSV at: {csv_path}")
    conn.close()


if __name__ == '__main__':
    MODEL_WEIGHT = '/home/diego/Documents/yolov7-tracker/mini_models/results/image_selection_model.pkl'
    SOLIDER_MODEL_PATH = '/home/diego/Documents/detectron2/solider_model.pth'
    ROOT_FOLDER = "/home/diego/Documents/yolov7-tracker/runs/detect/"
    
    
    RUN_FOLDER = 'bytetrack_santos_dumont' #ACA SE CAMBIA
    base_folder_images = 'imgs_santos_dumont'
    dest_folder_results = 'imgs_santos_dumont_top4'
    CSV_FILE = 'santos_dumont_bbox.db'
    FEAT_FILE = 'santos_dumont_features.csv'
    
    
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
        
        prepare_data_img_selection(db_path=DB_FILE_PATH, origin_table="bbox_raw", k_folds=4, n_images=5, new_table_name="bbox_img_selection")
        pbar.update(1)

        # 2.- Predict which images are good
        bbox_img_selection = predict_img_selection(db_file_path=DB_FILE_PATH, model_weights_path=MODEL_WEIGHT)
        pbar.update(1)

        # 3.- Apply image separation based on model results
        clean_img_folder_top_k(db_file_path=DB_FILE_PATH, base_folder_images=base_folder_images, dest_folder_results=dest_folder_results, k_fold=4, threshold=0.9)
        pbar.update(1)
        
        # Assuming `get_features_from_model` iterates over images, you could modify it to use tqdm internally or update here after completion
        # features = get_features_from_model(folder_path=dest_folder_results, weights=SOLIDER_MODEL_PATH, model='solider', features_file=features_file)
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
        

