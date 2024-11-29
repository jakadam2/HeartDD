import subprocess
import sys
import importlib.metadata
import time
import os
import re

ID_DESC_WEIGHTS = '1He7ELAxJM-RuKS9m4fOfvrtMmRuF-T4P'
ID_DETECTION_WEIGHTS = '1R74s94WH-X8VdCLCDlgb0jWxL0FLvb73'
ID_BINMASK = '1OFBtTf4SGWRGPSw0MOOyaF584q4iX2jC'


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


def download_gdrive_files():
    import gdown
    if not os.path.exists('server/description/weights/best.pth'):
        gdown.download(f'https://drive.google.com/uc?/export=download&id={ID_DESC_WEIGHTS}',
                       output='server/description/weights/best.pth')

    if not os.path.exists('server/detection/checkpoints/best.pt'):
        gdown.download(f'https://drive.google.com/uc?/export=download&id={ID_DETECTION_WEIGHTS}',
                       output='server/detection/checkpoints/best.pt')

    if not os.path.exists('base_images/good_df_newest.csv'):
        gdown.download(f'https://drive.google.com/uc?/export=download&id={ID_BINMASK}',
                       output='base_images/good_df_newest.csv')


def start_services():
    # Paths to the client and server scripts
    client_script = 'client/client.py'
    server_script = 'server/server.py'

    # Start the server process
    server_process = subprocess.Popen([sys.executable, server_script])
    print("Server started.")

    # Optional delay to allow the server to initialize
    time.sleep(1)

    # Start the client process
    client_process = subprocess.Popen([sys.executable, client_script])
    print("Client started.")

    # Wait for both processes to complete
    server_process.wait()
    client_process.wait()
    command = input("Type exit to exit the app")
    if command == "exit":
        print("exit was typed")
        # This doesn't work, I'm not sure why but it's not super important just Ctrl + C to kill the process


if __name__ == "__main__":
    download_dependencies()
    download_gdrive_files()
    start_services()
