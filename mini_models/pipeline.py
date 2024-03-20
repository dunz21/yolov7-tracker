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
from .re_ranking import complete_re_ranking, generate_re_ranking_html_report


def convert_csv_to_sqlite(csv_file_path, db_file_path, table_name='bbox_raw'):
    """
    Convert a CSV file to a SQLite table and return the data from the table.
    
    Parameters:
    - csv_file_path: The file path of the CSV to be converted.
    - db_file_path: The file path of the SQLite database.
    - table_name: The name of the table where the CSV data will be inserted. Defaults to 'bbox_data'.
    
    Returns:
    - A pandas DataFrame containing the data from the specified SQLite table.
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Create a connection to the SQLite database
    with sqlite3.connect(db_file_path) as conn:
        # Write the data to a SQLite table
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Fetch the newly inserted data to verify
        fetched_data = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    
    # Return the fetched data
    return fetched_data
    
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

def predict_img_selection(data, model_weights_path, save_csv_dir=None):
    """
    Predict image selection using a trained model and input data.
    
    Parameters:
    - data: A pandas DataFrame containing the input data for prediction.
    - model_weights_path: The file path to the trained model weights.
    - save_csv_dir: Optional directory path to save the prediction results as a CSV file.
    
    Returns:
    - A pandas DataFrame with the prediction results added.
    """
    features = ['area', 'centroid_x', 'centroid_y', 'frame_number', 'overlap', 'distance_to_center', 'conf_score']
    
    # Ensure the input data contains all the required features
    assert all(f in data.columns for f in features), "Some required columns for prediction are missing."
    
    # Load the model
    model = joblib.load(model_weights_path)
    
    # Perform predictions
    predictions = model.predict(data[features])
    predicted_confidences = model.predict_proba(data[features]).max(axis=1)
    
    # Add predictions to the DataFrame
    data['model_label_img'] = predictions
    data['model_label_conf'] = predicted_confidences
    data['model_label_conf'] = data['model_label_conf'].round(2)
    
    if save_csv_dir:
        # Save the modified DataFrame to a new CSV file
        predicted_csv_path = os.path.join(save_csv_dir, "img_selection_predicted.csv")
        data.to_csv(predicted_csv_path, index=False)
        print(f"Predictions saved to: {predicted_csv_path}")
    
    return data

def clean_img_folder_top_k(predict_csv, base_folder_images, dest_folder_results, k_fold, threshold=0.9):
    """
    Process and select images for re-identification based on predictions and k-fold cross-validation.
    
    Parameters:
    - predict_csv: Path to the CSV file containing predictions.
    - base_folder_images: Base folder path where images are stored.
    - dest_folder_results: Destination folder path where selected images will be stored.
    - k_fold: Number of folds for k-fold cross-validation.
    - threshold: Confidence threshold for selecting images.
    """
	# Load the predictions
    if isinstance(predict_csv, str):
        df = pd.read_csv(predict_csv)
    elif isinstance(predict_csv, pd.DataFrame):
        df = predict_csv
    else:
        raise ValueError("predict_csv must be a path to a CSV file or a pandas DataFrame")
    
    df['new_k_fold'] = None
    df['selected_image'] = False
    filtered_df = df.dropna(subset='img_name')
    filtered_df = filtered_df.sort_values(by=['id', 'frame_number'])
    
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder_results):
        os.makedirs(dest_folder_results)
    
    # Function to move selected images
    def copy_images(row):
        source_path = os.path.join(base_folder_images, row['img_name'].split('_')[1], row['img_name'])
        dest_path = source_path.replace(base_folder_images, dest_folder_results)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(source_path, dest_path)
    
    # Iterate over each unique id
    for id_value in filtered_df['id'].unique():
        local_threshold = threshold
        while True:
            id_df = filtered_df[filtered_df['id'] == id_value]
            filtered_id_df = id_df[(id_df['model_label_conf'] > local_threshold) & (id_df['model_label_img'] == 2)].copy()
            
            if len(filtered_id_df) >= k_fold or local_threshold <= 0:
                break
            local_threshold -= 0.05
        
        # If we have enough images, perform K-Fold and select one image per fold
        if len(filtered_id_df) >= k_fold:
            kf = KFold(n_splits=k_fold)
            
            for fold_number, (_, test_index) in enumerate(kf.split(filtered_id_df), start=1):
                fold_df = filtered_id_df.iloc[test_index]
                selected_row = fold_df.sample(n=1)
                selected_index = selected_row.index
                
                # Update the DataFrame with fold and selection information
                df.loc[selected_index, 'new_k_fold'] = fold_number
                df.loc[selected_index, 'selected_image'] = True
                
                # Move the selected image
                selected_row.apply(copy_images, axis=1)


if __name__ == '__main__':
    base_folder_images = '/home/diego/Documents/yolov7-tracker/imgs_santos_dumont/'
    dest_folder_results = '/home/diego/Documents/yolov7-tracker/imgs_santos_dumont_top4/'
    k_fold = 4
    threshold = 0.9
    
    model_weight = '/home/diego/Documents/yolov7-tracker/mini_models/results/image_selection_model.pkl'
    
    BASE_FOLDER_NAME = '/home/diego/Documents/yolov7-tracker/logs/'
    CSV_FILE = 'santos_dumont_bbox.csv'
    CSV_FILE_PATH = os.path.join(BASE_FOLDER_NAME, CSV_FILE)
    db_file_path = f'{BASE_FOLDER_NAME}/{CSV_FILE.replace(".csv", ".db")}'
    
    
    # 1.- To DB. Esto vendria despues listo
    
    # bbox_data = convert_csv_to_sqlite(CSV_FILE_PATH, db_file_path)
    
    # 2.- Predcit que imagenes son buenas
    # bbox_img_selection = predict_img_selection(bbox_data,model_weights_path=model_weight)
    # 3.- Aplicar separacion de imagenes basado en el resultado del modelo
    # clean_img_folder_top_k(bbox_img_selection, base_folder_images, dest_folder_results, k_fold, threshold)
    
    SOLIDER_MODEL_PATH = '/home/diego/Documents/detectron2/solider_model.pth'
    DISTANCE_METHOD = "cosine"
    features_file = f"/home/diego/Documents/yolov7-tracker/output/conce_solider_in-out_DB.csv"
    
    # features = get_features_from_model(folder_path=dest_folder_results,weights=SOLIDER_MODEL_PATH,model='solider',features_file=features_file)
    
    
    FRAME_RATE = 15
    n_images = 8
    max_number_back_to_compare = 57
    K1 = 8
    K2 = 3
    LAMBDA = 0
    save_csv_dir = '/home/diego/Documents/yolov7-tracker/output'

    
    
    with tqdm(total=6, desc="Overall Progress", unit="step") as pbar:
        # 1.- To DB. This would be ready afterwards
        bbox_data = convert_csv_to_sqlite(CSV_FILE_PATH, db_file_path)
        pbar.update(1)  # Update progress after each major step

        # 2.- Predict which images are good
        bbox_img_selection = predict_img_selection(bbox_data, model_weights_path=model_weight)
        pbar.update(1)

        # 3.- Apply image separation based on model results
        clean_img_folder_top_k(bbox_img_selection, base_folder_images, dest_folder_results, k_fold, threshold)
        pbar.update(1)
        
        # Assuming `get_features_from_model` iterates over images, you could modify it to use tqdm internally or update here after completion
        features = get_features_from_model(folder_path=dest_folder_results, weights=SOLIDER_MODEL_PATH, model='solider', features_file=features_file)
        pbar.update(1) 
        
        results, file_name = complete_re_ranking(features,
                                        n_images=n_images,
                                        max_number_back_to_compare=max_number_back_to_compare,
                                        K1=K1,
                                        K2=K2,
                                        LAMBDA=LAMBDA,
                                        filter_known_matches=None,
                                        save_csv_dir=save_csv_dir)
        pbar.update(1) 
        
        RE_RANK_HTML = os.path.join(save_csv_dir, f'{file_name}.html')
        generate_re_ranking_html_report(results, dest_folder_results, FRAME_RATE, RE_RANK_HTML)
        pbar.update(1)         
        

