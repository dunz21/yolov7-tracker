import os
import zipfile
import boto3
import shutil

def compress_files_in_folder(folder_path):
    # Define file paths for zip archives
    zip_files = {
        'imgs': os.path.join(folder_path, 'imgs.zip'),
        'rest': os.path.join(folder_path, 'rest.zip')
    }
    
    # Create zip files
    for zip_file in zip_files.values():
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            pass

    # List files and directories in the provided folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            # Process files
            if item.endswith('.csv'):
                zip_name = f"{item}.zip"
                zip_path = os.path.join(folder_path, zip_name)
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    zipf.write(item_path, item)
                os.remove(item_path)
            elif item.endswith('.db'):
                zip_name = f"{item}.zip"
                zip_path = os.path.join(folder_path, zip_name)
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    zipf.write(item_path, item)
                os.remove(item_path)
            elif not item.endswith(('.mkv', '.mp4', '.zip')):
                # Other files that should be compressed in 'rest.zip' without compression
                with zipfile.ZipFile(zip_files['rest'], 'a') as zipf:
                    zipf.write(item_path, item, compress_type=zipfile.ZIP_STORED)
                os.remove(item_path)
        elif os.path.isdir(item_path):
            # Process directories
            zip_dir(zip_files['imgs'], item_path)
            # Recursively delete directory after zipping
            shutil.rmtree(item_path)

    return folder_path

def zip_dir(zip_file_path, dir_path):
    with zipfile.ZipFile(zip_file_path, 'a') as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=dir_path)
                zipf.write(file_path, arcname)

def upload_results(local_path, s3_path, bucket_name):
    s3_client = boto3.client('s3')
    upload_success = True
    
    # Iterate over files in the local directory
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            
            # Construct the S3 key (path within the bucket)
            relative_path = os.path.relpath(local_file_path, local_path)
            s3_key = os.path.join(s3_path, relative_path)
            
            # Upload the file
            try:
                s3_client.upload_file(local_file_path, bucket_name, s3_key)
                print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
            except Exception as e:
                print(f"Failed to upload {local_file_path}: {e}")
                upload_success = False  # Mark upload as failed
                
    return upload_success

def delete_local_results_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder {folder_path} and all its contents have been deleted.")
        else:
            print(f"Folder {folder_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while trying to delete the folder: {e}")
                
                
def pipeline_compress_results_upload(folder_path, s3_path, bucket_name):
    compressed_folder = compress_files_in_folder(folder_path)
    success = upload_results(compressed_folder, s3_path, bucket_name)
    return success
