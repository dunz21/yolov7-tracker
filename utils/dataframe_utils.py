import pandas as pd
import random
import os

def filter_data(filename, id_col, num_samples=2):
    """
    Filter a DataFrame by randomly sampling a fixed number of rows for each unique ID.
    :param filename: Path to the CSV file
    :param id_col: Name of the column containing unique IDs
    :param num_samples: Number of samples to be taken for each unique ID
    :return: Filtered DataFrame
    """

    df = pd.read_csv(filename)
    filtered_data = pd.DataFrame()  # Initialize an empty DataFrame to store the results
    unique_ids = df[id_col].unique()  # Get unique IDs

    for id in unique_ids:
        df_id = df[df[id_col] == id]  # Filter rows for each ID
        if len(df_id) > num_samples:
            df_id_sample = df_id.sample(num_samples)  # Randomly select 2 rows
        else:
            df_id_sample = df_id  # If less than 2 rows, take all rows
        filtered_data = pd.concat([filtered_data, df_id_sample])

    return filtered_data

def keep_two_random_images(main_folder, number_images=2):
    # Iterate through all subfolders in the main folder
    for folder_name in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, folder_name)

        if os.path.isdir(subfolder_path):
            # List all image files in the subfolder
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if len(image_files) > number_images:
                # Randomly select number_images images to keep
                images_to_keep = random.sample(image_files, number_images)

                # Delete all other images
                for image in image_files:
                    if image not in images_to_keep:
                        os.remove(os.path.join(subfolder_path, image))
                print(f"Processed subfolder: {folder_name}")
            else:
                print(f"Skipped subfolder with less than or equal to {number_images} images: {folder_name}")

# Example usage
keep_two_random_images("imgs_conce")