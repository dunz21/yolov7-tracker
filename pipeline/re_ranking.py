import numpy as np
import torch
import datetime
import os
import base64
import pandas as pd
from scipy.spatial.distance import cdist
from collections import Counter
from tqdm import tqdm
import sqlite3
from utils.time import seconds_to_time
from pipeline.vit_pipeline import get_files
from utils.types import Direction
from pipeline.vit_pipeline import get_features_from_model

def number_to_letters(num):
    mapping = {i: chr(122 - i) for i in range(10)}
    num_str = str(num)
    letter_code = ''.join(mapping[int(digit)] for digit in num_str)
    return letter_code

def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat = None, only_local = False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        # print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)
        original_dist = distmat.numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

#     print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def eval_simplified_with_matches(distmat, q_pids, g_pids):
    indices = np.argsort(distmat, axis=1)  # Sorted indices of gallery samples for each query, axis=1 means columns so horizontally
    # q_pids[:, np.newaxis] == q_pids.reshape(4,1) or better q_pids.reshape(-1,1)
    matchs = np.hstack((q_pids[:, np.newaxis], g_pids[indices]))
    
    dist_matrix_sort = np.hstack((q_pids[:, np.newaxis], np.sort(distmat, axis=1)))
    return np.dstack((matchs, dist_matrix_sort))

def find_repeating_values(input_list, min_repeats=3):
    frequency = Counter(input_list)
    repeating_values = [value for value, count in frequency.items() if count >= min_repeats]
    if repeating_values:
        return repeating_values[0]
    else:
        return False
    
## 1. Load data
def load_data(features_csv):
    
    # Check if re_ranking_data is a path (string) or a DataFrame and load accordingly
    if isinstance(features_csv, str):
        features_df = pd.read_csv(features_csv)
    elif isinstance(features_csv, pd.DataFrame):
        features_df = features_csv
    else:
        raise ValueError("features_df must be a path to a CSV file or a pandas DataFrame")

    # Read features and convert to floats
    for col in features_df.columns[3:]:
        features_df[col] = features_df[col].astype(float)
    
    # Extracting ids, img_names, and directions for filtering
    ids = features_df['id'].values
    img_names = features_df['img_name'].values
    directions = features_df['direction'].values
    
    # Convert features to tensor and normalize
    feature_tensor = torch.tensor(features_df.iloc[:, 3:].values, dtype=torch.float32)
    feature_tensor = feature_tensor / feature_tensor.norm(dim=1, keepdim=True)

    return ids, img_names, directions, feature_tensor
## 2. Process re-ranking
def process_re_ranking(ids, img_names, directions, feature_tensor, n_images=4, max_number_back_to_compare=60, K1=8, K2=3, LAMBDA=0.1,matches=None,autoeval=False):
    results_dict = {}  # Initialize as a dictionary
    posible_pair_matches, ids_correct_ins,out_without_in = [], np.array([]),[]
    id_in_list = np.unique(ids[directions == Direction.In.value])
    id_out_list = np.unique(ids[directions == Direction.Out.value])

    for id_out in tqdm(id_out_list, desc="Processing Re Rank IDs"):
        if id_out < id_in_list[0]:
            continue
        
        query_indices = np.where((ids == id_out) & (directions == Direction.Out.value))[0]
        query = feature_tensor[query_indices]
        q_pids = img_names[query_indices]

        ## Esto me sirve cuando es alimentado por la BD de matches en el labeler
        if matches:
            ids_correct_ins = []
            all_id_except_match_out = [matches.get(key_out) for key_out in matches.keys() if key_out != id_out]
            ids_correct_ins = np.append(ids_correct_ins, all_id_except_match_out)


        gallery_candidate_indices = np.where((ids < id_out) & (directions == Direction.In.value) & (~np.isin(ids, ids_correct_ins)))[0]
        if len(gallery_candidate_indices) == 0:
            out_without_in.append(id_out)
            continue
        if len(gallery_candidate_indices) > max_number_back_to_compare:
            gallery_candidate_indices = gallery_candidate_indices[-max_number_back_to_compare:]

        gallery = feature_tensor[gallery_candidate_indices]
        g_pids = img_names[gallery_candidate_indices]

        distmat = re_ranking(query, gallery, K1, K2, LAMBDA)
        matching_gallery_ids = eval_simplified_with_matches(distmat, q_pids, g_pids)

        if autoeval:
            rank1_list = [int(m.split('_')[1]) for m in matching_gallery_ids[:,1,0]]
            rank1_match = find_repeating_values(rank1_list)
            ### ojo que dentro de la RANK1 pueden haber 3 malos pero 1 bueno, y ese bueno va a ser la distancia mas cercana
            ### Otra forma de hacer esto es tener [img_id_1,img_id_1,img_id_1,img_id_2] y solo obtener la distancia pero de los que se parecen
            ### Pero esta forma simple es mas rapida. Despues podria probar la otra y comparar. Por ahora lo importante es llegar a algo
            near_distance = np.sort([float(value) for value in matching_gallery_ids[:,1,1]])[0]
            if rank1_match and near_distance < 0.6:
                posible_pair_matches.append([int(id_out),int(rank1_match)])
                ids_correct_ins = np.append(ids_correct_ins, rank1_match)
        results_dict[id_out] = matching_gallery_ids[:,:n_images + 1]

    return results_dict,posible_pair_matches
## 3. Save results
def save_re_ranking(results_list='', posible_pair_matches='',db_path='',FRAME_RATE=15):
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        def format_value(tuple,query):
            query_frame_number = int(query.split('_')[2])
            img_name, distance = tuple
            distance = np.round(float(distance),decimals=2) if img_name != query else ''
            img_frame_number = int(img_name.split('_')[2])
            video_time = seconds_to_time((int(img_name.split('_')[2])// FRAME_RATE))
            time = seconds_to_time(max(0,(query_frame_number - img_frame_number)) // FRAME_RATE)
            return {
                'id': f"{img_name.split('_')[1]}_{number_to_letters(img_name.split('_')[2])}",
                'image_path': f"{img_name.split('_')[1]}/{img_name}.png",
                'time': time,
                'video_time': video_time,
                'distance': distance
            }
        
        def format_row(arr):
            new_list = []
            for row in arr:
                query = row[0][0]
                new_list.append([format_value(value,query) for value in row])
            return new_list
        
        list_out = {}
        for id_out in results_list:
            list_out[str(id_out)] = format_row(results_list[id_out])
            
        cursor.execute('''DROP TABLE IF EXISTS reranking''')
        cursor.execute('''DROP TABLE IF EXISTS reranking_matches''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reranking_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_in INTEGER NOT NULL,
                id_out INTEGER NOT NULL UNIQUE,
                count_matches INTEGER,
                obs TEXT,
                ground_truth BOOLEAN DEFAULT 1
            )
            ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reranking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_out TEXT,
                img_out TEXT,
                id_in TEXT,
                img_in TEXT,
                time_diff TEXT,
                video_time TEXT,
                distance TEXT,
                rank INTEGER
            )
        ''')
        
        for id_out, top_k_list in list_out.items():
            for row_gallery in top_k_list:
                rank = 1  # Initialize rank for each id_out group
                cursor.execute('''
                        INSERT INTO reranking (id_out, img_out, id_in, img_in, time_diff, video_time, distance, rank)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (id_out, row_gallery[0]['image_path'], None, None, None, row_gallery[0]['video_time'], None, None))
                for item in row_gallery[1:]:
                    img_out = row_gallery[0]['image_path']
                    id_in = item['id'].split('_')[0]
                    img_in = item['image_path']
                    time_diff = item['time']
                    video_time = item['video_time']
                    distance = str(item['distance'])  # Ensure distance is a string for consistency
                    cursor.execute('''
                        INSERT INTO reranking (id_out, img_out, id_in, img_in, time_diff, video_time, distance, rank)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (id_out, img_out, id_in, img_in, time_diff, video_time, distance, rank))

                    rank += 1  # Increment rank for each row

        conn.commit()
        # Initialize an empty dictionary for the results
        predict_matches = {}
        for out_in_tuple in posible_pair_matches:
            id_out = out_in_tuple[0]
            id_in = out_in_tuple[1]
            predict_matches[id_out] = {
                'id_in': id_in
            }
            
            # Attempt to insert or update
            try:
                cursor.execute('''
                INSERT INTO reranking_matches (id_in, id_out)
                VALUES (?, ?)
                ON CONFLICT(id_out) DO UPDATE SET
                    id_in = excluded.id_in,
                    count_matches = excluded.count_matches,
                    obs = excluded.obs
                ''', (id_in, id_out))
                conn.commit()
            except sqlite3.IntegrityError as e:
                print('Error:', e)
            
        conn.close() 
        print('Re-ranking results saved successfully!')

    except Exception as e:
        print('Error:', e)

        
def complete_re_ranking(features_csv, n_images=4, max_number_back_to_compare=60, K1=8, K2=3, LAMBDA=0.1, db_path=''):
    ids, img_names, directions, feature_tensor = load_data(features_csv)
    results_list, posible_pair_matches = process_re_ranking(ids, img_names, directions, feature_tensor, n_images, max_number_back_to_compare, K1, K2, LAMBDA, autoeval=True)
    save_re_ranking(results_list, posible_pair_matches, db_path=db_path, FRAME_RATE=15)

def generate_re_ranking_html_report(re_ranking_data, base_folder, frame_rate, re_rank_html):
    def _image_formatter(image_name, query_frame_number):
        folder_id = image_name.split('_')[1]
        img_path = os.path.join(base_folder, str(folder_id), f"{image_name.split('|')[0]}.png")
        distance = image_name.split('|')[1]
        try:
            img_frame_number = int(image_name.split('_')[2])
            with open(img_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode()
                time = seconds_to_time(max(0,(query_frame_number - img_frame_number)) // frame_rate)
                video_time = seconds_to_time((int(image_name.split('_')[2])// frame_rate))
                html_distance = f'<div>Distance: {distance}</div>' if distance != '-' else ''
                return f'<div><img width="125" src="data:image/png;base64,{encoded_string}">{html_distance}<div>ID: {image_name.split("_")[1]}_{number_to_letters(image_name.split("_")[2])} - {time} </div><div>{video_time}</div></div>'
        except OSError as e:
            return f"OSError: {e}, File: {img_path}"


    # Check if re_ranking_data is a path (string) or a DataFrame and load accordingly
    if isinstance(re_ranking_data, str):
        re_ranking = pd.read_csv(re_ranking_data)
    elif isinstance(re_ranking_data, pd.DataFrame):
        re_ranking = re_ranking_data
    else:
        raise ValueError("re_ranking_data must be a path to a CSV file or a pandas DataFrame")

    df = re_ranking.copy()
    # df['IndexImg'] = re_ranking.groupby('query').cumcount() + 1
    df['frame_number_query'] = df['query'].apply(lambda x: int(x.split('_')[2]))

    for column in df.columns.drop('frame_number_query'):
        df[column] = df.apply(lambda x: _image_formatter(x[column],x['frame_number_query']), axis=1)

    html_df = df.drop(['frame_number_query'],axis=1).to_html(escape=False, index=False)

    with open(re_rank_html, 'w') as file:
        file.write(html_df)
        
if __name__ == '__main__':
    
    files = get_files('/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_26_calper_portugal')
    db = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_26_calper_portugal/calper_portugal_bbox.db'
    imgs = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_26_calper_portugal/imgs_calper_portugal'
    csv = '/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_26_calper_portugal/calper_portugal_bbox'
    FRAME_RATE = 15
    n_images = 8
    max_number_back_to_compare = 57
    K1 = 8
    K2 = 3
    LAMBDA = 0
    # filter_known_matches = '/home/diego/Desktop/MatchSimple.csv'  
    # filter_known_matches = None

    
    
    convert_csv_to_sqlite(csv_file_path=f"{csv}.csv", db_file_path=db, table_name='bbox_raw')
    switch_id_corrector_pipeline(db_path=db, base_folder_path=imgs,weights='model_weights.pth',model_name='solider')
    prepare_data_img_selection(db_path=db, origin_table="bbox_raw", k_folds=4, n_images=5, new_table_name="bbox_img_selection")
    predict_img_selection(db_file_path=db, model_weights_path='mini_models/results/image_selection_model.pkl')
    clean_img_folder_top_k(db_file_path=db, base_folder_images=imgs, dest_folder_results=f"{imgs}_top4", k_fold=4, threshold=0.9)
    features = get_features_from_model(model_name='solider', folder_path=f"{imgs}_top4", weights='/home/diego/Documents/yolov7-tracker/model_weights.pth', db_path=db)
    complete_re_ranking(features,n_images=8,max_number_back_to_compare=57,K1=8,K2=3,LAMBDA=0,db_path=db)
    