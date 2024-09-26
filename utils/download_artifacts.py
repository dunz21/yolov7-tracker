import boto3

def download_files_from_s3():
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

    # Loop through each file and download from S3
    for file in files:
        print(f"Downloading {file}...")
        # Download the file from S3 to the current directory
        s3.download_file(bucket_name, file, file)

    print("Download completed.")

if __name__ == "__main__":
    download_files_from_s3()
