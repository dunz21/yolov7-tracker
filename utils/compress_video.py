import subprocess
import os
import time
import logging

def compress_and_replace_video(video_path, encoder='h264_nvenc', preset='slow', cq=40):
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

def print_mkv_files_in_subfolders(folder_path):
    try:
        # Check if the given path is a valid directory
        if not os.path.isdir(folder_path):
            print(f"The path {folder_path} is not a valid directory.")
            return
        
        # Iterate over all items in the folder path
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            
            # Check if the item is a directory
            if os.path.isdir(subfolder_path):
                # Iterate over all items in the subfolder
                for item in os.listdir(subfolder_path):
                    item_path = os.path.join(subfolder_path, item)
                    
                    # Check if the item is an .mkv file
                    if os.path.isfile(item_path) and item_path.endswith('.mkv'):
                        file_size_gb = os.path.getsize(item_path) / (1024 ** 3) # Convert size to GB
                        # Print only if file size is greater than 5GB
                        if file_size_gb > 5:
                            print(f"File: {item_path}, Size: {file_size_gb:.2f} GB")
                            compress_and_replace_video(item_path)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
def compress_for_video_viewer_folder_root(folder_path):
    try:
        # Check if the given path is a valid directory
        if not os.path.isdir(folder_path):
            print(f"The path {folder_path} is not a valid directory.")
            return
        
        # Iterate over all items in the folder path
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # Check if the item is an .mkv file
            if os.path.isfile(item_path) and item_path.endswith('.mkv'):
                file_size_gb = os.path.getsize(item_path) / (1024 ** 3) # Convert size to GB
                
                # Print only if file size is greater than 5GB
                if file_size_gb > 5:
                    print(f"File: {item_path}, Size: {file_size_gb:.2f} GB")
                    new_item_path = item_path.replace('.mkv', '_compressed.mp4')
                    ffmpeg_command = [
                        'ffmpeg',
                        '-i', item_path,
                        "-vf", "scale=1280:720",
                        "-c:v", "h264_nvenc",
                        "-cq:v", '40',
                        "-preset", 'slow',
                        "-c:a", "copy",  
                        new_item_path,
                        '-y',  # Overwrite output files without asking
                    ]
                    # Run the ffmpeg command
                    subprocess.run(ffmpeg_command, check=True)
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# folder_path = "/home/diego/mydrive/results/1/10/8/"
# print_mkv_files_in_subfolders(folder_path)


if __name__ == "__main__":
    # Example usage
    # video_path = "/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_26_calper_portugal/portugal_20240424.mp4"
    # compress_and_replace_video(video_path)
    
    
    # Compres for video viewer
    folder_path = "/home/diego/mydrive/footage/1/10/8/"
    compress_for_video_viewer_folder_root(folder_path)








def process_videos_to_15_FPS(folder_path):
    # List all files in the specified folder
    all_files = os.listdir(folder_path)

    for filename in all_files:
        # Check if the file is a video (MP4 or MKV) and doesn't have 'FPS' in its name
        if (filename.endswith(".mp4") or filename.endswith(".mkv")) and "FPS" not in filename:
            # Construct full file path
            input_file = os.path.join(folder_path, filename)
            
            # Determine output file name
            file_root, file_ext = os.path.splitext(filename)
            output_file = os.path.join(folder_path, f"{file_root}_FPS{file_ext}")
            
            # Check if the processed file already exists
            if not os.path.exists(output_file):
                # Construct the ffmpeg command
                command = [
                    "ffmpeg",
                    "-y",  # Overwrite output files without asking
                    "-i", input_file,
                    "-vf", "fps=15",
                    "-c:v", "h264_nvenc",
                    "-preset", "fast",
                    "-b:v", "1M",
                    output_file
                ]
                
                # Execute the ffmpeg command
                subprocess.run(command)

#process_videos('/home/diego/mydrive/footage/1/12/2/')