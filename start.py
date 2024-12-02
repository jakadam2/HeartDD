import subprocess
import sys
import importlib.metadata
from importlib.resources import files
import time
import os
import re




def download_dependencies():
    """ This script downloads all the dependencies in requirements.txt. Therefore it needs to be here before any non
    standard imports"""
    try:  # Read requirements from the file
        with open('requirements.txt') as f:
            required_packages = f.read().splitlines()
            # Get the list of packages already installed
            package_names = [re.split(r'[~=<>!]+', pkg)[0].lower() for pkg in required_packages]

            installed_packages = {dist.metadata['Name'].lower() for dist in importlib.metadata.distributions()}
            # Identify packages that are not installed
            missing_packages = [pkg for pkg in package_names if pkg not in installed_packages]
            if missing_packages:
                print(f"Installing missing packages: {', '.join(missing_packages)}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            else:
                print("All dependencies are already installed.")

    except FileNotFoundError as e:
        print(f"Error: requirements.txt file not found. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while checking/installing dependencies: {e}")
        sys.exit(1)





def start_services():
    # Paths to the client and server scripts
    client_script = files('hdd.client') / 'client.py'
    server_script = files('hdd.server') / 'server.py'

    # Start the server process
    server_process = subprocess.Popen([sys.executable, server_script])
    print("Server starting...")

    # Optional delay to allow the server to initialize
    time.sleep(2)

    # Start the client process
    client_process = subprocess.Popen([sys.executable, client_script])
    print("Client starting...")

    # Wait for both processes to complete
    server_process.wait()
    client_process.wait()
    command = input("Type exit to exit the app")
    if command == "exit":
        print("exit was typed")
        # This doesn't work, I'm not sure why but it's not super important just Ctrl + C to kill the process


if __name__ == "__main__":
    download_dependencies()
    start_services()
