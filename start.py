import subprocess
import sys
import importlib.metadata
import time
import os
import re
import configparser

parser = configparser.ConfigParser()
config = parser.read("config.ini")

ID_DESC_WEIGHTS = parser.get("google.downloads","ID_DESC_WEIGHTS")
DESC_WEIGHTS_DIR = parser.get("google.downloads","DESC_WEIGHTS_DIR")

ID_DETECTION_WEIGHTS = parser.get("google.downloads","ID_DETECTION_WEIGHTS")
DETECTION_WEIGHTS_DIR = parser.get("google.downloads","DETECTION_WEIGHTS_DIR")

ID_BINMASK = parser.get("google.downloads","ID_BINMASK")
BINMASK_DIR = parser.get("google.downloads","BINMASK_DIR")

GOOGLE_DOWNLOAD_LINK = "https://drive.google.com/uc?/export=download&id="

def download_dependencies():
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
    if not os.path.exists(DESC_WEIGHTS_DIR):
        gdown.download(f'https://drive.google.com/uc?/export=download&id={ID_DESC_WEIGHTS}',
                       output=DESC_WEIGHTS_DIR)

    if not os.path.exists(DETECTION_WEIGHTS_DIR):
        gdown.download(f'https://drive.google.com/uc?/export=download&id={ID_DETECTION_WEIGHTS}',
                       output=DETECTION_WEIGHTS_DIR)

    if not os.path.exists(BINMASK_DIR):
        gdown.download(f'https://drive.google.com/uc?/export=download&id={ID_BINMASK}',
                       output=BINMASK_DIR)

def start_services(mode=None):
    # Paths to the client and server scripts
    client_script = 'client/start_client.py'
    server_script = 'server/server.py'

    try:
        # Start the server process
        server_process = subprocess.Popen([sys.executable, server_script])
        print("Server starting...")
        #Server needs a bit of time to start, that's why we sleep
        time.sleep(2)

        #Client can be run in two modes, normal mode or bulk mode
        client_command = [sys.executable, client_script]
        if mode == "bulk":
            client_command.append("bulk")  # Add the bulk argument if provided

        client_process = subprocess.Popen(client_command)
        print("Client starting...")

        while True:
            # Check if the client process has terminated
            if client_process.poll() is not None:
                print("Client process has exited.")
                break
            time.sleep(1)  # Wait a bit before checking again

    finally:
        # Shut down the server process
        print("Shutting down the server...")
        server_process.terminate()
        server_process.wait()
        print("Server shut down.")


if __name__ == "__main__":
    download_dependencies()
    download_gdrive_files()
    if len(sys.argv) > 1:
        start_services(sys.argv[1])
    else:
        start_services()
