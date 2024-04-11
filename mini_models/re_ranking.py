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
from utils.tools import seconds_to_time, number_to_letters

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
    id_in_list = np.unique(ids[directions == 'In'])
    id_out_list = np.unique(ids[directions == 'Out'])

    for id_out in tqdm(id_out_list, desc="Processing Re Rank IDs"):
        if id_out < id_in_list[0]:
            continue
        
        query_indices = np.where((ids == id_out) & (directions == 'Out'))[0]
        query = feature_tensor[query_indices]
        q_pids = img_names[query_indices]

        ## Esto me sirve cuando es alimentado por la BD de matches en el labeler
        if matches:
            ids_correct_ins = []
            all_id_except_match_out = [matches.get(key_out) for key_out in matches.keys() if key_out != id_out]
            ids_correct_ins = np.append(ids_correct_ins, all_id_except_match_out)


        gallery_candidate_indices = np.where((ids < id_out) & (directions == 'In') & (~np.isin(ids, ids_correct_ins)))[0]
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
def save_results(results_dict, K1, K2, LAMBDA, n_images, filter_known_matches, save_csv_dir):
    # Preparing the data for DataFrame
    data_for_df = []
    for query_id, matches in results_dict.items():
        for row in matches:
            # Suma el nombre de la imagen con el valor de la distancia
            new_join_row = row[:,0] +'|'+ np.append(['-'],np.around(row[1:,1].astype(float), decimals=2))
            data_for_df.append(new_join_row)

    # Defining column names dynamically based on n_images
    column_names = ['query'] + [f'rank{i}' for i in range(1, n_images + 1)]
    # Creating DataFrame
    re_ranking_results = pd.DataFrame(data_for_df, columns=column_names)

    file_name = f're_ranking_k1_{K1}_k2_{K2}_lambda_{LAMBDA}_num_img_{n_images}_{"filtered" if filter_known_matches else "all"}'
    if save_csv_dir:
        CSV_FILE_PATH = os.path.join(save_csv_dir, f'{file_name}.csv')
        re_ranking_results.to_csv(CSV_FILE_PATH, index=False)

    return re_ranking_results, file_name

def classification_match(posible_pair_matches='', filename_csv='',db_path=''):
    # Initialize the results
    results = {'total_rows': 0, 'total_matched': 0}
    
    # if len(posible_pair_matches) > 0:
    #     # Convert the input matches to a DataFrame and save to CSV
    #     df = pd.DataFrame(posible_pair_matches, columns=['id_out', 'id_in'])
    #     if filename_csv != '':
    #         df.to_csv(filename_csv, index=False)
        
    #     conn = sqlite3.connect(db_path)
    #     df.to_sql('auto_match', conn, if_exists='replace', index=False)
        
    #     results['total_rows'] = len(df)
    #     (conn.cursor()).execute('''
    #         CREATE TABLE IF NOT EXISTS reranking_matches (
    #             id INTEGER PRIMARY KEY AUTOINCREMENT,
    #             id_in INTEGER NOT NULL,
    #             id_out INTEGER NOT NULL UNIQUE,
    #             count_matches INTEGER,
    #             obs TEXT,
    #             ground_truth BOOLEAN DEFAULT 1
    #         )
    #     ''')
        #### RE DO!!!
        # # Approach 2: Using CTE to find total matched
        # query_cte = """
        # WITH Matched AS (
        #     SELECT am.*
        #     FROM auto_match am
        #     INNER JOIN reranking_matches rm ON am.id_out = rm.id_out AND am.id_in = rm.id_in
        # )
        # SELECT COUNT(*) as total_matched FROM Matched;
        # """
        # results['total_matched'] = pd.read_sql_query(query_cte, conn).iloc[0]['total_matched']
        
        # # Close the database connection
        # conn.close()
    
    return results
        
def complete_re_ranking(features_csv, n_images=4, max_number_back_to_compare=60, K1=8, K2=3, LAMBDA=0.1, save_csv_dir=None):
    ids, img_names, directions, feature_tensor = load_data(features_csv)
    results_list, posible_pair_matches = process_re_ranking(ids, img_names, directions, feature_tensor, n_images, max_number_back_to_compare, K1, K2, LAMBDA, autoeval=True)
    re_ranking_results, file_name = save_results(results_list, K1, K2, LAMBDA, n_images, None, save_csv_dir)
    return re_ranking_results, file_name, posible_pair_matches

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
    ROOT_FOLDER = '/home/diego/Documents/tracklab/yolov7-tracker/runs/detect/2024_04_03_conce_debug_switch_id/'
    BASE_FOLDER = '/home/diego/Documents/tracklab/yolov7-tracker/runs/detect/2024_04_03_conce_debug_switch_id/imgs_conce_debug_top4'
    FRAME_RATE = 15
    n_images = 8
    max_number_back_to_compare = 57
    K1 = 8
    K2 = 3
    LAMBDA = 0
    # filter_known_matches = '/home/diego/Desktop/MatchSimple.csv'  
    # filter_known_matches = None
    save_csv_dir = '/home/diego/Documents/yolov7-tracker/logs'

    conn = sqlite3.connect('/home/diego/Documents/tracklab/yolov7-tracker/runs/detect/2024_04_03_conce_debug_switch_id/conce_debug_bbox.db')
    cursor = conn.cursor()    
    features_csv = pd.read_sql('SELECT * FROM features', conn)
    
    
    results, file_name, posible_pair_matches = complete_re_ranking(features_csv,
                                            n_images=n_images,
                                            max_number_back_to_compare=max_number_back_to_compare,
                                            K1=K1,
                                            K2=K2,
                                            LAMBDA=LAMBDA,
                                            )
    # print(posible_pair_matches)
    # # Complete
    # # RE_RANK_HTML = os.path.join(save_csv_dir, f'{file_name}.html')

    # # generate_re_ranking_html_report(results, BASE_FOLDER, FRAME_RATE, RE_RANK_HTML)
    # final_classifier = classification_match(posible_pair_matches=posible_pair_matches,filename_csv=f"{ROOT_FOLDER}/auto_match.csv",db_path=f"{ROOT_FOLDER}/santos_dumont_bbox.db")
    # print(final_classifier)
    