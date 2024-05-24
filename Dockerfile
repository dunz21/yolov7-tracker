# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 as base

# Set the working directory
WORKDIR /app

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Install Python and pip
RUN apt-get install -y python3 python3-pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the application
CMD ["python3", "detect_and_track.py"]