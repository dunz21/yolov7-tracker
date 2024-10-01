import vastai
import os

def destroy_instance_from_env():
    # Get VAST_CONTAINERLABEL from the environment variable
    container_label = os.getenv('VAST_CONTAINERLABEL')
    if not container_label:
        print("VAST_CONTAINERLABEL environment variable is not set.")
        return
    
    # Extract the instance ID
    instance_id = container_label.split('.')[1]
    
    # Initialize Vast.ai CLI
    client = vastai.Client()

    # Destroy the instance
    try:
        result = client.destroy_instance(instance_id)
        print(f"Instance {instance_id} destroyed successfully.")
        return result
    except Exception as e:
        print(f"Error destroying instance: {e}")
        return None