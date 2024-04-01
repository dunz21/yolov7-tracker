# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 as base

# Set noninteractive installation mode to avoid getting prompted
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    build-essential \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to make python and pip commands point to python3 and pip3 respectively
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .
RUN pip install numpy
# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set the default command for the container. Using `tail -f /dev/null` to keep the container running indefinitely, so you can exec into it.
CMD ["tail", "-f", "/dev/null"]
