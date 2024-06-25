import os
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm


# Clean up 1: Remove IDs with low area
def list_ids_to_remove_based_on_area(df, plot=False,bins_number=10):
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(df['area'], bins=bins_number, edgecolor='black')
        plt.title('Histogram of Area')
        plt.xlabel('Area')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
  

    hist, bin_edges = np.histogram(df['area'], bins=bins_number)

    # Find the bin with the highest frequency
    max_bin_index = np.argmax(hist)
    most_frequent_bin_start = bin_edges[max_bin_index]
    most_frequent_bin_end = bin_edges[max_bin_index + 1]

    # Filter the DataFrame to get the rows that fall into the most frequent bin
    most_frequent_areas = df[(df['area'] >= most_frequent_bin_start) & (df['area'] < most_frequent_bin_end)]

    # Get the unique IDs associated with these areas
    unique_ids = most_frequent_areas['id'].unique()
    filtered_results = df[~df['id'].isin(unique_ids)]
    print('Number of IDs based on area:', len(unique_ids))
    return filtered_results, unique_ids

# Clean up 2: Remove IDs with low movement
def list_ids_to_remove_based_on_movement(df, plot_histogram=False, threshold=100):
    df = df.sort_values(by=['id', 'frame_number'])

    # Function to calculate the distance between first and last centroid
    def calculate_distance(group):
        if len(group) > 1:
            first_x, first_y = group.iloc[0][['centroid_x', 'centroid_y']]
            last_x, last_y = group.iloc[-1][['centroid_x', 'centroid_y']]
            distance = np.sqrt((last_x - first_x) ** 2 + (last_y - first_y) ** 2)
        else:
            distance = 0
        return distance
    
    # Calculate the total movement for each ID
    total_movement_df = df.groupby('id').apply(calculate_distance).reset_index()
    total_movement_df.columns = ['id', 'total_movement']
    
    if plot_histogram:
        # Visualize the distribution of total movement
        plt.figure(figsize=(10, 6))
        plt.hist(total_movement_df['total_movement'], bins=30, edgecolor='black')
        plt.title('Distribution of Total Movement per ID')
        plt.xlabel('Total Movement')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    
    # Filter IDs with total movement below the threshold
    low_movement_ids = total_movement_df[total_movement_df['total_movement'] < threshold]['id']
    list_ids_to_remove = low_movement_ids.to_list()
    
    # Filter the original DataFrame to remove these IDs
    filtered_df = df[~df['id'].isin(low_movement_ids)]
    print('Number of IDs based on movement:', len(list_ids_to_remove))
    return filtered_df, list_ids_to_remove

# Clean up 1 and 2 combined: Remove IDs with low area and low movement
def get_new_csv_clean_up_csv_based_on_area_movement(old_csv='',new_csv='', plot_area=False, plot_movement=False, threshold_movement=100):
    df = pd.read_csv(old_csv)
    filtered_df_area, ids_to_remove_area = list_ids_to_remove_based_on_area(df, plot=plot_area)
    filtered_df_movement, ids_to_remove_movement = list_ids_to_remove_based_on_movement(filtered_df_area, plot_histogram=plot_movement, threshold=threshold_movement)
    print('Number of IDs area:', len(ids_to_remove_area), 'Number of IDs movement:', len(ids_to_remove_movement))
    filtered_df_movement.to_csv(new_csv, index=False)
    return filtered_df_movement


# FOR DEBUG IMAGES THAT I WILL DROP, ONLY AVAIALBE ON NOTEBOOK
def view_imgs_by_list_id(list_ids, img_path_folder, img_size=(50, 50), grid_columns=20):
    images = []
    ids = []
    
    for img_id in list_ids:
        id_folder = os.path.join(img_path_folder, str(img_id))
        if not os.path.exists(id_folder):
            continue
        
        img_files = os.listdir(id_folder)
        if not img_files:
            continue
        
        # Randomly pick one image from the folder
        img_file = random.choice(img_files)
        img_file_path = os.path.join(id_folder, img_file)
        
        # Read and resize the image
        img = cv2.imread(img_file_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
        
        images.append(img)
        ids.append(img_id)
    
    # Determine the grid size
    grid_rows = len(images) // grid_columns + 1
    
    # Create the plot
    fig, axes = plt.subplots(grid_rows, grid_columns, figsize=(grid_columns, grid_rows))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Plot the images
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
            ax.set_title(str(ids[i]), fontsize=8)
        else:
            ax.axis('off')
    
    plt.show()

# Generate images by the CSV file
def generate_img_by_bbox(csv_path, video_path, img_path, skip_frames=3, show_progress=True):
    # Load CSV data
    df = pd.read_csv(csv_path)
    max_frame_number = df['frame_number'].max()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Create the base folder for generated images
    img_generated_path = os.path.join(img_path, 'img_generated')
    os.makedirs(img_generated_path, exist_ok=True)
    
    current_frame = 0
    if show_progress:
        progress_bar = tqdm(total=max_frame_number, desc="Processing frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame > max_frame_number:
            break
        
        if current_frame % skip_frames == 0:
            if show_progress:
                progress_bar.update(skip_frames)
            
            # Process each row in the DataFrame for the current frame
            frame_data = df[df['frame_number'] == current_frame]
            
            for _, row in frame_data.iterrows():
                img_id = row['id']
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                
                # Extract the image using bounding box coordinates
                cropped_img = frame[y1:y2, x1:x2]
                
                # Create a directory for the current id if it doesn't exist
                id_folder = os.path.join(img_generated_path, str(img_id))
                os.makedirs(id_folder, exist_ok=True)
                
                # Save the image
                img_save_path = os.path.join(id_folder, row['img_name'])
                cv2.imwrite(img_save_path, cropped_img)
        
        current_frame += 1
    
    cap.release()
    if show_progress:
        progress_bar.close()
    print("Image extraction completed!")

if __name__ == '__main__':
    results = pd.read_csv('/home/diego/mydrive/results/1/3/1/tobalaba_entrada_20240604_1000/tobalaba_entrada_20240604_1000_bbox.csv')

    initial_list_ids = results['id'].unique()
    print('Initial number of IDs:', len(initial_list_ids))

    df,list_ids = list_ids_to_remove_based_on_area(results, plot=False)
    print('Number of IDs after removing low area IDs:', len(df['id'].unique()),'and IDs were removed:', len(list_ids))

    total_movement_df,list_ids_to_remove  = list_ids_to_remove_based_on_movement(df, plot_histogram=False, threshold=100)

    print('Number of IDs after removing low movement IDs:', len(total_movement_df['id'].unique()), 'and IDs were removed:', len(list_ids_to_remove))