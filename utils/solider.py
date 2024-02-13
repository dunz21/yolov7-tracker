import torch
from torchvision import transforms
from PIL import Image
import glob
import os
import numpy as np
import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler



def plot_mds(features_array="", image_names=[],simpleLegend=True, title="", figsize=(12,10), scaler=False):
    if len(features_array) == 0:
        print(f"Sin gráfico por no tener valores para {title}")
        return False
    if scaler:
        scaler = StandardScaler().fit(features_array)
        features_array = scaler.transform(features_array)
    # Apply MDS
    mds = MDS(n_components=2)
    mds_result = mds.fit_transform(features_array)
    
    # Extract prefix and suffix from image names for coloring
    prefixes = [int(name.split('_')[1]) for name in image_names]
    suffixes = [int(name.split('_')[2]) for name in image_names]
    max_suffix = max(suffixes)
    min_alpha = 0.6
    normalized_suffixes = [min_alpha + (1 - min_alpha) * (s / max_suffix) for s in suffixes]
    
    # Create a mapping of prefix to palette index
    unique_prefixes = list(set(prefixes))
    prefix_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes)}
    added_to_legend = set()
    legend_handles_labels = []

    # Plotting
    plt.figure(figsize=figsize)
    palette = sns.color_palette("husl", len(unique_prefixes))
    for i, (x, y) in enumerate(mds_result):
        color = palette[prefix_to_index[prefixes[i]]]
        label = None
        if simpleLegend:
            if prefixes[i] not in added_to_legend:
                label = f"img_{prefixes[i]}"
                added_to_legend.add(prefixes[i])
        else:
            label = f"{image_names[i].split('_')[3][0]}{prefixes[i]}_{suffixes[i]}"
            added_to_legend.add(prefixes[i])
        plt.text(x, y, f"{prefixes[i]}_{suffixes[i]}", fontsize=8, ha='right', va='bottom')
        handle = plt.scatter(x, y, color=(color[0], color[1], color[2], normalized_suffixes[i]), label=label)

        if label:
            legend_handles_labels.append((handle, label))

    # Sort the handles and labels
    legend_handles_labels = sorted(legend_handles_labels, key=lambda x: x[1])
    sorted_handles, sorted_labels = zip(*legend_handles_labels)

    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.title(f"MDS {title}")
    plt.legend(handles=sorted_handles, labels=sorted_labels)
    plt.show()

def preprocess_images(img_paths, height, width):
    """
    Process a list of image paths to a batch of images ready for model inference.
    """
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = [transform(Image.open(img_path).convert('RGB')) for img_path in img_paths]
    return torch.stack(images)  # Stack images into a single tensor

def extract_images_from_subfolders(folder_paths):
    # If the input is a string (single folder path), convert it into a list
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
    
    all_images = []
    
    for folder_path in folder_paths:
        # Walk through each main folder and its subfolders
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # For each subfolder, find all .png images
            images = glob.glob(os.path.join(dirpath, '*.png'))
            all_images.extend(images)
    return all_images

def solider_result(folder_path="", soldier_weight=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = torch.load(soldier_weight)
    loaded_model.eval().to(device)

    images = extract_images_from_subfolders(folder_path)
    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]

    # Process images to tensor
    images_tensor = preprocess_images(images, 384, 128).to(device)
    
    with torch.no_grad():
        features_list, _ = loaded_model(images_tensor)
    
    features_array = features_list.cpu().numpy()
    return features_array, image_names

def img_to_feature(images_path=[],solider_weight=''):
    loaded_model = torch.load(solider_weight)
    loaded_model.eval()  # Set the model to evaluation mode
    # Extract image names from paths

    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images_path]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    # Extract features
    total_batch = [torch.stack([preprocess_image(img,384,128)], dim=0) for img in images_path]
    with torch.no_grad():
        features_list, _ = loaded_model(torch.cat(total_batch,dim=0).to(device))
    
    features_array = features_list.cpu().numpy()
    return features_array, image_names

def in_out_status(values, number_img_different_permited=2):
    """
        Determines if the values in a list are all the same, different, or mixed.
        number_img_different_permited: number of images that can be different
    """
    in_count = values.count('In')
    out_count = values.count('Out')

    # If all values are the same, return that value
    if in_count == len(values) or out_count == len(values):
        return values[0]

    # si existe 10 In y 2 Out entonces que sea In, pero si hay 3 Out ya no
    if abs(in_count - out_count) >= len(values) - (2 * number_img_different_permited):
        return 'In' if in_count > out_count else 'Out'

    # If there are two or more elements that differ, return 'InOut'
    return 'InOut'

def custom_threshold_analysis(arr, threshold=75):

    sorted_arr = np.sort(arr)
    lowest = sorted_arr[0]
    second_lowest = sorted_arr[1]
    percentage_diff = (lowest / second_lowest) * 100

    is_considerably_lower = percentage_diff < threshold
    return {
        "lowest_value": lowest,
        "second_lowest": second_lowest,
        "percentage_difference": percentage_diff,
        "is_considerably_lower": is_considerably_lower
    }

def seconds_to_time(seconds):
    # Create a timedelta object
    td = datetime.timedelta(seconds=seconds)
    # Add the timedelta to a minimal datetime object
    time = (datetime.datetime.min + td).time()
    # Convert to a string format
    return time.strftime("%H:%M:%S")

def plot_mds_dbscan(features_array="", image_names=[], plot=True, title="", figsize=(12, 10), eps=0.5, min_samples_ratio=0.15, min_include=3, scaler=True):
    if scaler:
        scaler = StandardScaler().fit(features_array)
        features_array = scaler.transform(features_array)
    # Apply MDS
    mds = MDS(n_components=2, random_state=42)
    mds_result = mds.fit_transform(features_array)
    
    count_image_cluster = pd.DataFrame({'images': image_names,'id': [img.split('_')[1] for img in image_names]}).groupby('id').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    idcluster1 , sizecluster1 = count_image_cluster.iloc[0,0] , count_image_cluster.iloc[0,1]
    idcluster2 , sizecluster2 = count_image_cluster.iloc[1,0] , count_image_cluster.iloc[1,1]
    

    # Apply DBSCAN clustering
    min_samples = int(sizecluster1*min_samples_ratio)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(mds_result)
    labels = db.labels_

    
    ### DATA FRAME ####
    data_images = pd.DataFrame({'images': image_names,'id': [img.split('_')[1] for img in image_names],'labels': db.labels_})
    count_data = data_images[data_images.labels != -1].groupby('labels').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    if len(count_data) == 0:
        return False, ''
    id_biggest_cluster_size = count_data.iloc[0,0]
    overlap_images = data_images[data_images.labels == id_biggest_cluster_size].groupby('id').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    if len(overlap_images) > 1:
        if overlap_images.iloc[1,1] > min_include:
            total_images_inside_big_cluster = ', '.join([f"ID: {row[0]} Total: {row[1]}" for index,row in overlap_images.iloc[1:].reset_index(drop=True).iterrows()])
            msg = f"Total de imagenes {total_images_inside_big_cluster} encontradas en cluster ID: {idcluster1}  min_samples: {min_samples}"
            print(msg)
    ### DATA FRAME ####

    if plot:
        # Define a color palette for DBSCAN clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # excluding noise
        cluster_palette = sns.color_palette('husl', n_clusters)
        # Handle noise in the data
        colors = [(0.5, 0.5, 0.5) if label == -1 else cluster_palette[label] for label in labels]
        
        # Plotting
        plt.figure(figsize=figsize)
        for i, (x, y) in enumerate(mds_result):
            plt.text(x, y, f"{image_names[i]}", fontsize=8, ha='right', va='bottom')
            plt.scatter(x, y, color=colors[i], label=f'Cluster {labels[i]}' if labels[i] != -1 else 'Noise')

        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.title(f"MDS/ DBSCAN eps {eps} min_samples {min_samples} {title}")
        
        # Create a legend for the clusters
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_palette[i], markersize=10) for i in range(n_clusters)]
        labels = [f'Cluster {i}' for i in range(n_clusters)]
        if -1 in labels:  # if there's noise
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.5, 0.5, 0.5), markersize=10))
            labels.append('Noise')
        plt.legend(handles=handles, labels=labels)
        plt.show()