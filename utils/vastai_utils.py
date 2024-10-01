import subprocess
import os

def destroy_instance():
    """
    Destroys the current Vast.ai instance based on the VAST_CONTAINERLABEL environment variable.
    """

    # Retrieve the VAST_CONTAINERLABEL environment variable
    container_label = os.getenv('VAST_CONTAINERLABEL')
    if not container_label:
        print("Error: VAST_CONTAINERLABEL environment variable is not set.")
        return
    
    # Attempt to extract the instance ID
    try:
        instance_id = container_label.split('.')[1]
    except IndexError:
        print(f"Error: Could not parse instance ID from VAST_CONTAINERLABEL: {container_label}")
        return
    
    # Define the vast.ai destroy command
    command = ["vastai", "destroy", "instance", instance_id]

    # Execute the command
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Output: {result.stdout.strip()}")
        print(f"Instance {instance_id} destroyed successfully.")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error destroying instance: {e.stderr.strip()}")
        return None
    except Exception as ex:
        print(f"Unexpected error occurred: {ex}")
        return None


if __name__ == "__main__":
    destroy_instance()