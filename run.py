"""
Main Entry Point for Pneumonia Detection Application
Initializes and runs the GUI application
"""

import tkinter as tk
from gui import PneumoniaApp

if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaApp(root)
    root.mainloop()


