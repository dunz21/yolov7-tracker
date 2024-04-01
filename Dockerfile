# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 as base

# Set noninteractive installation mode, to avoid getting prompted
ENV DEBIAN_FRONTEND=noninteractive

# Install essential tools and build essentials
RUN apt-get update && apt-get install -y wget git libgl1-mesa-glx build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install essential tools
# RUN apt-get update && apt-get install -y wget git libgl1-mesa-glx && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV MINICONDA_VERSION=py39_4.9.2
RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh  && \
    sh /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Add Conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Set the default command for the container. This is what gets executed when you run the container without specifying a command.
# Using `tail -f /dev/null` to keep the container running indefinitely, so you can exec into it.
CMD ["tail", "-f", "/dev/null"]
