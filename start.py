import subprocess
import sys
import importlib.metadata
import time

def install_missing_dependencies():
    try:
        # Read requirements from the file
        with open('requirements.txt') as f:
            required_packages = f.read().splitlines()

        # Get the list of packages already installed
        installed_packages = {dist.metadata['Name'].lower() for dist in importlib.metadata.distributions()}

        # Identify packages that are not installed
        missing_packages = [
            pkg for pkg in required_packages if pkg.lower().split("==")[0] not in installed_packages
        ]

        if missing_packages:
            print(f"Installing missing packages: {', '.join(missing_packages)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
        else:
            print("All dependencies are already installed.")
    except FileNotFoundError:
        print("Error: requirements.txt file not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while checking/installing dependencies: {e}")
        sys.exit(1)

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

if __name__ == "__main__":
    install_missing_dependencies()  # Step 1: Check for and install missing dependencies
    start_services()                # Step 2: Start client and server
