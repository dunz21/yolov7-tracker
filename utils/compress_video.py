import subprocess
import os
import time
import logging

def compress_and_replace_video(video_path, encoder='h264_nvenc', preset='fast', cq=40):
    """
    Compresses a video using FFmpeg, replaces the original video with the compressed version.

    Parameters:
        video_path (str): Path to the video file to be compressed and replaced.
        encoder (str): The video encoder to use. Default is 'h264_nvenc' for NVIDIA GPU acceleration.
        preset (str): Preset for the video encoder. Affects the balance between processing speed and compression efficiency.
        cq (int): Constant quality setting for the encoder. Lower values mean better quality.
    """
    t0 = time.time()
    logger = logging.getLogger(__name__)
    video_path = os.path.abspath(video_path)
    # Generate the path for the temporary compressed video
    compressed_path = f"{video_path.rsplit('.', 1)[0]}_compressed.mp4"

    # FFmpeg command for compressing the video
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        "-c:v", encoder,
        "-preset", preset,
        "-cq:v", str(cq),
        "-c:a", "copy",  # Keep the audio stream unchanged
        compressed_path
    ]

    try:
        # Execute the compression command
        subprocess.run(ffmpeg_command, check=True)
        print("Compression successful.")

        # Replace the original video file with the compressed video
        os.replace(compressed_path, video_path)
        print("Original video replaced with the compressed version.")
        
    except subprocess.CalledProcessError:
        print("Error during video compression. FFmpeg command failed.")
    except OSError as e:
        print(f"Error handling files: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
        
    logger.info(f"Video compression took {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    # Example usage
    video_path = "/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_26_calper_portugal/portugal_20240424.mp4"
    compress_and_replace_video(video_path)