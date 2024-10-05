import os
import boto3
import subprocess
import shutil
from utils.types import QueueVideoStatus
import time


def get_video_duration(video_file):
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1', video_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = float(result.stdout.strip())
    return duration

def compress_and_upload_video(video_file, client_id, store_id, channel_camera_id, progress_callback=None):
    # Define directories and paths
    footage_dir = os.path.dirname(video_file)
    bucket_name = os.getenv('VIDEOS_MIVO_BUCKET_NAME')
    compressed_dir = os.path.join(footage_dir, 'compress_video')
    s3_key = f"video-viewer/{client_id}/{store_id}/{channel_camera_id}/{os.path.basename(video_file)}"

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Check if file already exists in S3
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File {s3_key} already exists in S3. Skipping upload.")
        return 0,0
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            print(f"Error checking S3: {e}")
            return 0,0

    # Create compressed directory if it does not exist
    os.makedirs(compressed_dir, exist_ok=True)

    # Define output path for compressed video
    compressed_video_file = os.path.join(compressed_dir, os.path.basename(video_file))

    # Compress video using ffmpeg with progress
    ffmpeg_command = [
        'ffmpeg', '-i', video_file, '-vf', 'scale=1280:720',
        '-c:v', 'h264_nvenc', '-preset', 'fast', '-cq:v', '40', '-c:a', 'copy',
        '-progress', '-', '-nostats',
        compressed_video_file
    ]

    print(f"Compressing video: {video_file}")
    start_compress_time = time.time()
    try:
        total_duration = get_video_duration(video_file)
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1) as process:
            for line in process.stdout:
                if 'out_time_ms=' in line:
                    time_str = line.strip().split('=')[1]
                    out_time_ms = int(time_str)
                    progress = (out_time_ms / (total_duration * 1000000)) * 100
                    if progress_callback:
                        progress_callback(progress, QueueVideoStatus.VIDEO_ENCODE.value)
                    print(f"Encoding progress: {progress:.2f}%", end='\r')
        print(f"\nSuccessfully compressed: {compressed_video_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error compressing video: {e}")
        return 0,0
    time_video_encoding = time.time() - start_compress_time
    # Upload compressed video to S3 with progress
    print(f"Uploading {compressed_video_file} to S3 bucket {bucket_name}")
    try:
        file_size = os.path.getsize(compressed_video_file)
        bytes_transferred = 0

        def upload_progress(bytes_amount):
            nonlocal bytes_transferred
            bytes_transferred += bytes_amount
            progress = (bytes_transferred / file_size) * 100
            if progress_callback:
                progress_callback(progress, QueueVideoStatus.UPLOADING_VIDEO_ENCODE.value)
            print(f"Upload progress: {progress:.2f}%", end='\r')
        start_upload_video = time.time()
        s3.upload_file(
            compressed_video_file,
            bucket_name,
            s3_key,
            Callback=upload_progress
        )
        print(f"\nSuccessfully uploaded to S3: {compressed_video_file}")
        time_video_upload = time.time() - start_upload_video
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return 0,0

    # Cleanup: remove the temporary compressed directory
    try:
        shutil.rmtree(compressed_dir)
        print(f"Temporary directory {compressed_dir} deleted.")
    except Exception as e:
        print(f"Error deleting temporary directory: {e}")
    return time_video_encoding, time_video_upload