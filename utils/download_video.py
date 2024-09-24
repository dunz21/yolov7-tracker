import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


def find_video_in_s3(s3_bucket, s3_path, date):
    # Validate date format YYYYMMDD
    if len(date) != 8 or not date.isdigit():
        raise ValueError("Date must be in format YYYYMMDD.")

    s3_client = boto3.client('s3')

    try:
        # List objects in the bucket with the specified prefix
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_path)
    except NoCredentialsError:
        print("AWS credentials not available.")
        return False, None
    except ClientError as e:
        print(f"Error accessing S3: {e}")
        return False, None

    # Search for the video that matches the date
    for obj in response.get('Contents', []):
        if date in obj['Key']:
            full_s3_video_path = obj['Key']
            print(f"Found video {full_s3_video_path} in S3.")
            return True, full_s3_video_path

    print(f"No video found in S3 for date {date}.")
    return False, None

class DownloadProgress:
    def __init__(self, file_size):
        self._file_size = file_size
        self._bytes_downloaded = 0

    def __call__(self, bytes_amount):
        self._bytes_downloaded += bytes_amount
        progress_percentage = (self._bytes_downloaded / self._file_size) * 100
        print(f"Download progress: {progress_percentage:.2f}%", end='\r')

def download_video_from_s3(s3_bucket, path_to_download_video, full_s3_video_path):
    s3_client = boto3.client('s3')

    try:
        # Get the size of the file from S3
        response = s3_client.head_object(Bucket=s3_bucket, Key=full_s3_video_path)
        file_size = response['ContentLength']

        os.makedirs(os.path.dirname(path_to_download_video), exist_ok=True)

        # Create an instance of DownloadProgress to track download progress
        progress = DownloadProgress(file_size)

        # Download the file with progress callback
        s3_client.download_file(
            s3_bucket, 
            full_s3_video_path, 
            path_to_download_video, 
            Callback=progress
        )

        print(f"\nVideo downloaded to {path_to_download_video}.")
    except NoCredentialsError:
        print("AWS credentials not available.")
    except ClientError as e:
        print(f"Error downloading video from S3: {e}")
        
def delete_local_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted.")
        else:
            print(f"File {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while trying to delete the file: {e}")

# # Usage Example
# s3_bucket = 'your-bucket-name'
# s3_path = 'client123/store456/camera789/'
# date = '20240907'
# path_to_download_video = '/home/diego/mydrive/footage/4/19/2/leonisa_parquearauco_entrada_20240907_1000_C_3min.mkv'

# # Find the video in S3
# found, full_s3_video_path = find_video_in_s3(s3_bucket, s3_path, date)

# # If found, download the video
# if found:
#     download_video_from_s3(path_to_download_video, full_s3_video_path)
# else:
#     print("Video not found, download skipped.")
