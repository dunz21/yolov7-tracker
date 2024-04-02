# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
    

# Copy the current directory contents (server folder) into the container
COPY server/ /app/

# Copy the rest of the needed files from the root directory
# This assumes you have a requirements.txt for your dependencies in the root
# And that all other dependencies can be encapsulated
COPY reid /app/reid
COPY utils /app/utils
COPY mini_models /app/mini_models

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 3001

# Define environment variable
# ENV NAME World

# Run server.py when the container launches
CMD ["python", "./server.py"]
