import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


def find_video_in_s3(s3_bucket, s3_path, date, exact=True):
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
        # Extract the file name excluding the path and date
        key = obj['Key']
        if exact:
            # If exact match is required, ensure that only the date part is variable
            full_s3_video_path = obj['Key']
            print(f"Found exact video {full_s3_video_path} in S3.")
            return True, full_s3_video_path
        else:
            # If not exact, check if the date is part of the key
            if date in key:
                full_s3_video_path = obj['Key']
                print(f"Found video {full_s3_video_path} in S3.")
                return True, full_s3_video_path

    print(f"No video found in S3 for date {date}.")
    return False, None

class DownloadProgress:
    def __init__(self, file_size, progress_callback=None, progress_interval=5):
        self._file_size = file_size
        self._bytes_downloaded = 0
        self.progress_callback = progress_callback
        self.progress_interval = progress_interval
        self._last_reported_percentage = 0

    def __call__(self, bytes_amount):
        self._bytes_downloaded += bytes_amount
        progress_percentage = (self._bytes_downloaded / self._file_size) * 100
        if progress_percentage - self._last_reported_percentage >= self.progress_interval:
            self._last_reported_percentage = progress_percentage
            if self.progress_callback:
                self.progress_callback(progress_percentage)
        print(f"Download progress: {progress_percentage:.2f}%", end='\r')

def download_video_from_s3(s3_bucket, path_to_download_video, full_s3_video_path, progress_callback=None, progress_interval=5):
    s3_client = boto3.client('s3')

    try:
        response = s3_client.head_object(Bucket=s3_bucket, Key=full_s3_video_path)
        file_size = response['ContentLength']

        os.makedirs(os.path.dirname(path_to_download_video), exist_ok=True)

        progress = DownloadProgress(file_size, progress_callback, progress_interval)

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

