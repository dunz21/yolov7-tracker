import cv2
import numpy as np
from tools.solider import in_out_status, custom_threshold_analysis
from utils.tools import seconds_to_time
from tools.PersonImage import PersonImage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from detectron2.utils.logger import setup_logger
import logging

class PersonImageComparer:
    list_in = []
    list_out = []
    list_result = []
    list_banner_in = None
    offset_overlay_in = 0
    list_banner_out = None
    offset_overlay_out = 0
    max_width = 1920

    list_total_features = []
    silhouette_avg = pd.DataFrame()
    pair_row_col = {
        'row': [],
        'col': [],
        'value': []
    }

    # @classmethod
    # def process_person_image(cls, person_image: PersonImage):
    #     list_in_out = [p.split('_')[3] for p in person_image.list_images]
    #     in_or_out = in_out_status(list_in_out,2)
    #     if in_or_out != "InOut":
    #         person_image.list_features, person_image.list_names = img_to_feature(person_image.list_images)
    #         for image_name, features in zip(person_image.list_names, person_image.list_features):
    #             id = image_name.split('_')[1]
    #             direction = in_or_out
    #             row_data = [image_name, id, direction]
    #             row_data.extend(features)
    #             cls.list_total_features.append(row_data)
            
            

    #     if in_or_out == "In":
    #         if not any(p.id == person_image.id for p in cls.list_in):
    #             cls.list_in.append(person_image)
    #             cls.add_image_to_banner(person_image.list_images[0], in_or_out)
    #             logging.info('IN So will this')
    #     elif in_or_out == "Out":
    #         if not any(p.id == person_image.id for p in cls.list_out):
    #             logging.info('OUT So will this')
    #             cls.list_out.append(person_image)
    #             cls.compare_and_process()
    #             cls.add_image_to_banner(person_image.list_images[0], person_image.direction)
            
    @classmethod
    def compare_and_process(cls):
        #CONSTRUCCTION OF SILHOUETTE AVG
        for image_out in cls.list_out:
            for image_in in cls.list_in:
                features_out = [x[3:] for x in cls.list_total_features if x[1] == str(image_out.id)]
                features_in = [x[3:] for x in cls.list_total_features if x[1] == str(image_in.id)]
                data_cluster = np.vstack((features_out, features_in))
                kmeans = KMeans(n_clusters=2, random_state=42).fit(data_cluster)
                if len(data_cluster) > 1:
                    unique_labels = np.unique(kmeans.labels_)
                    if 1 < len(unique_labels) < len(data_cluster):
                        score = silhouette_score(data_cluster, kmeans.labels_)
                        if image_in.id not in cls.silhouette_avg.index:
                            cls.silhouette_avg = cls.silhouette_avg.reindex(cls.silhouette_avg.index.tolist() + [image_in.id])
                        if image_out.id not in cls.silhouette_avg.columns:
                            cls.silhouette_avg[image_out.id] = np.nan
                        cls.silhouette_avg.at[image_in.id, image_out.id] = score
        # PROCESO DE MATCH
        DELETE_PREV_ROWS = True
        DELETE_PREV_COL = True
        THRESHOLD = 93
        for col in cls.silhouette_avg.columns:
            col_data = np.array(cls.silhouette_avg[col].dropna())

            if col_data.size == 0:
                continue

            # CASO BORDE
            if len(col_data) == 1:
                if np.min(col_data) < 0.33:
                    thereshold = {
                        "lowest_value": np.min(col_data),
                        "second_lowest": 0,
                        "percentage_difference": 0,
                        "is_considerably_lower": True
                    }
                else: 
                    thereshold = {
                        "lowest_value": np.min(col_data),
                        "second_lowest": 0,
                        "percentage_difference": 0,
                        "is_considerably_lower": False
                    }
            else:
                thereshold = custom_threshold_analysis(col_data, threshold=THRESHOLD)

            if thereshold['is_considerably_lower']:
                # Estoy casi seguro que es el mejor
                min_value_col = np.min(col_data)
                index_row = cls.silhouette_avg[cls.silhouette_avg[col] == min_value_col].index[0]
                cls.pair_row_col['row'].append(index_row)
                cls.pair_row_col['col'].append(col)
                cls.pair_row_col['value'].append(min_value_col)
                if DELETE_PREV_ROWS:
                    cls.silhouette_avg = cls.silhouette_avg.drop(index_row)
                if DELETE_PREV_COL:
                    cls.silhouette_avg = cls.silhouette_avg.drop(col, axis=1)
                print(cls.pair_row_col)
                
    @classmethod
    def find_time_diff(cls, person_image_out: PersonImage):
        FRAME_RATE = 15
        try:
            find_row_index = cls.pair_row_col['col'].index(person_image_out.id)
        except:
            return None
        index_row_person_in = cls.pair_row_col['row'][find_row_index]
        person_in = next((obj for obj in cls.list_in if obj.id == index_row_person_in), None)
        diff_time = int(person_image_out.list_images[0].split('_')[2]) - int(person_in.list_images[0].split('_')[2]) // FRAME_RATE
        return seconds_to_time(diff_time)


    @classmethod
    def put_diff_time_to_img(cls,img,diff_time="00:01:00"):
        # Choose font, size, color, and thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        color = (0,255,0)  # White color
        thickness = 1
        # Get the size of the text box
        (text_width, text_height), baseline = cv2.getTextSize(diff_time, font, font_scale, thickness)

        # Position text at the bottom of the image
        text_x = (img.shape[1] - text_width) // 2  # Center the text
        text_y = img.shape[0] - 10  # 10 pixels from the bottom
        cv2.putText(img, diff_time, (text_x, text_y), font, font_scale, color, thickness)

    @classmethod
    def add_image_to_banner(cls, path, direction):
        fixed_width, fixed_height = 50, 100
        img = cv2.imread(path)
        if img is not None:
            resized_img = cv2.resize(img, (fixed_width, fixed_height))


        if direction == "In":
            if cls.list_banner_in is None:
                cls.list_banner_in = resized_img
            else:
                if cls.list_banner_in.shape[1] + resized_img.shape[1] <= cls.max_width:
                    cls.list_banner_in = np.hstack((cls.list_banner_in, resized_img))
                else:
                    if cls.offset_overlay_in + resized_img.shape[1] >= cls.max_width:
                        cls.offset_overlay_in = 0
                    cls.list_banner_in[:, cls.offset_overlay_in:cls.offset_overlay_in+resized_img.shape[1], :] = resized_img
                    cls.offset_overlay_in += resized_img.shape[1]
        elif direction == "Out":
            time = cls.find_time_diff(cls.list_out[-1])
            if time is not None:
                cls.put_diff_time_to_img(img=resized_img, diff_time=time)
            if cls.list_banner_out is None:
                cls.list_banner_out = resized_img
            else:
                if cls.list_banner_out.shape[1] + resized_img.shape[1] <= cls.max_width:
                    cls.list_banner_out = np.hstack((cls.list_banner_out, resized_img))
                else:
                    if cls.offset_overlay_out + resized_img.shape[1] >= cls.max_width:
                        cls.offset_overlay_out = 0
                    cls.list_banner_out[:, cls.offset_overlay_out:cls.offset_overlay_out+resized_img.shape[1], :] = resized_img
                    cls.offset_overlay_out += resized_img.shape[1]


