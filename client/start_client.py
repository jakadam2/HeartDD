import sys
from client import Client
from bulk_client import BulkClient
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

def start():
    if len(sys.argv) > 1:
        flag = sys.argv[1]
        if flag != "bulk":
            return
        client = BulkClient()
        client.bulk_operations()
    else:
        root = tk.Tk()
        Client(root)
        root.mainloop()


if __name__ == "__main__":
    start()