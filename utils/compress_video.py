import subprocess
import os

def compress_and_replace_video(video_path, scale='1280:720', encoder='h264_nvenc', preset='slow', cq=28):
    """
    Compresses a video using FFmpeg, replaces the original video with the compressed version.

    Parameters:
        video_path (str): Path to the video file to be compressed and replaced.
        scale (str): The target resolution for scaling the video. Default is '1280:720'.
        encoder (str): The video encoder to use. Default is 'h264_nvenc' for NVIDIA GPU acceleration.
        preset (str): Preset for the video encoder. Affects the balance between processing speed and compression efficiency.
        cq (int): Constant quality setting for the encoder. Lower values mean better quality.
    """
    video_path = os.path.abspath(video_path)
    # Generate the path for the temporary compressed video
    compressed_path = f"{video_path.rsplit('.', 1)[0]}_compressed.mp4"

    # FFmpeg command for compressing the video
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"scale={scale}",
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


if __name__ == "__main__":
    # Example usage
    video_path = "/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_17_conce_debug/test_tracker_1.mp4"
    compress_and_replace_video(video_path, scale='1280:720', encoder='h264_nvenc', preset='slow', cq=28)