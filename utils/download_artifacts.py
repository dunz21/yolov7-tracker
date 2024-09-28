import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

class DownloadProgress:
    def __init__(self, file_size):
        self._file_size = file_size
        self._bytes_downloaded = 0

    def __call__(self, bytes_amount):
        self._bytes_downloaded += bytes_amount
        progress_percentage = (self._bytes_downloaded / self._file_size) * 100
        print(f"Download progress: {progress_percentage:.2f}%", end='\r')

def check_and_download_files_from_s3():
    # Define the S3 bucket name and file paths
    bucket_name = "artifacts-mivo"
    files = [
        "model_weights.pth",
        "transformer_120.pth",
        "yolo_persons.v5_f012_yolov10m-pc.pt",
        "yolov7.pt"
    ]

    # Initialize the S3 client
    s3 = boto3.client('s3')

    # Check and download missing files
    for file in files:
        if os.path.isfile(file):
            print(f"{file} already exists, skipping download.")
        else:
            print(f"{file} not found, downloading...")

            try:
                # Get the size of the file from S3
                response = s3.head_object(Bucket=bucket_name, Key=file)
                file_size = response['ContentLength']

                # Create an instance of DownloadProgress to track download progress
                progress = DownloadProgress(file_size)

                # Download the file with progress callback
                s3.download_file(
                    bucket_name, 
                    file, 
                    file, 
                    Callback=progress
                )

                print(f"\n{file} downloaded successfully.")

            except NoCredentialsError:
                print("AWS credentials not available.")
                return
            except ClientError as e:
                print(f"Error downloading {file} from S3: {e}")
                return

    print("All files checked and downloaded if necessary.")

if __name__ == "__main__":
    check_and_download_files_from_s3()
