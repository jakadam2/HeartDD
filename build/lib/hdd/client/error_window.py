try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

class ErrorWindow():
    @staticmethod
    def show(title:str = "Unknown Error", message: str = "Unknown Error"):
        tk.messagebox.showerror(title = title, message = message)
