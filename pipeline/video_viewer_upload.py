import os
import boto3
import subprocess
import shutil

def compress_and_upload_video(video_file, client_id, store_id, channel_camera_id):
    # Define directories and paths
    footage_dir = os.path.dirname(video_file)
    compressed_dir = os.path.join(footage_dir, 'compress_video')
    s3_bucket = f"s3://videos-mivo/video-viewer/{client_id}/{store_id}/{channel_camera_id}/"
    s3_key = f"{client_id}/{store_id}/{channel_camera_id}/{os.path.basename(video_file)}"

    # Initialize S3 client
    s3 = boto3.client('s3')
    bucket_name = "videos-mivo"

    # Check if file already exists in S3
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File {s3_key} already exists in S3. Skipping upload.")
        return
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            print(f"Error checking S3: {e}")
            return

    # Create compressed directory if it does not exist
    os.makedirs(compressed_dir, exist_ok=True)

    # Define output path for compressed video
    compressed_video_file = os.path.join(compressed_dir, os.path.basename(video_file))

    # Compress video using ffmpeg
    ffmpeg_command = [
        'ffmpeg', '-i', video_file, '-vf', 'scale=1280:720', 
        '-c:v', 'h264_nvenc', '-preset', 'fast', '-cq:v', '40', '-c:a', 'copy',
        compressed_video_file
    ]
    
    print(f"Compressing video: {video_file}")
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Successfully compressed: {compressed_video_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error compressing video: {e}")
        return

    # Upload compressed video to S3
    print(f"Uploading {compressed_video_file} to S3 bucket {s3_bucket}")
    try:
        s3.upload_file(compressed_video_file, bucket_name, s3_key)
        print(f"Successfully uploaded to S3: {compressed_video_file}")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return

    # Cleanup: remove the temporary compressed directory
    try:
        shutil.rmtree(compressed_dir)
        print(f"Temporary directory {compressed_dir} deleted.")
    except Exception as e:
        print(f"Error deleting temporary directory: {e}")
