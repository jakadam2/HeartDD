import subprocess
import time

# Define the paths to the client and server scripts
client_script = 'client/client.py'
server_script = 'server/server.py'

# Start the server process
server_process = subprocess.Popen(['python', server_script])
print("[MAIN] Server started.")

# Optional delay to allow server to initialize before starting the client
time.sleep(2)

# Start the client process
client_process = subprocess.Popen(['python', client_script])
print("[MAIN] Client started.")

# Wait for both processes to complete
server_process.wait()
client_process.wait()
