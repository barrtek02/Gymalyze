import tkinter as tk
from tkinter import PhotoImage
from tkinter import ttk

import sv_ttk

from gui.login_screen import LoginScreen
from gui.home_screen import HomeScreen
from gui.live_detection import LiveDetectionScreen
from gui.train_exercise import TrainSpecificScreen
from gui.upload_review import UploadVideoScreen
from gui.sign_up_screen import SignUpScreen
from utils.database import Database  # Your database class
import os


class BodybuildingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.set_window_geometry()
        self.set_app_icon()
        self.db = Database()

        # Container for holding frames (screens)
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Dictionary to hold all screens (frames)
        self.frames = {}
        self.create_frames()

        # Start by showing the login screen
        self.show_frame("LoginScreen")

        sv_ttk.set_theme("dark")  # Set the theme to dark
        # Optional: Handle resizing globally for all screens
        self.bind("<Configure>", self.on_resize)

    def set_window_geometry(self):
        """Set the initial window geometry and handle resizing."""
        # Get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Set window to 50% of screen size and center it
        window_width = int(screen_width * 0.5)
        window_height = int(screen_height * 0.5)
        x_position = int((screen_width - window_width) / 2)
        y_position = int((screen_height - window_height) / 2)

        self.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Set a minimum window size to prevent layout breakage
        self.minsize(400, 300)  # Minimum window size of 400x300

    def set_app_icon(self):
        """Set the app icon using the gymalyze icon from assets"""
        icon_path = os.path.join(
            os.path.dirname(__file__), "assets", "gymalyze_icon.png"
        )
        if os.path.exists(icon_path):
            self.icon_img = PhotoImage(file=icon_path)
            self.iconphoto(False, self.icon_img)  # Set the image as the icon

    def create_frames(self):
        """Initialize and store all frames in a dictionary for easy access."""
        for F in (
            LoginScreen,
            HomeScreen,
            LiveDetectionScreen,
            TrainSpecificScreen,
            UploadVideoScreen,
            SignUpScreen,
        ):
            frame = F(
                parent=self.container, controller=self, db=self.db
            )  # Pass the shared db connection
            frame_name = F.__name__  # Get the name of the class as a string
            self.frames[frame_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Configure resizing properties for the container
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

    def show_frame(self, frame_name):
        """Show the desired frame by bringing it to the front."""
        frame = self.frames[frame_name]
        frame.tkraise()  # Raise the selected frame to the front

    def on_resize(self, event):
        """Optional: Handle resizing logic."""
        width = event.width
        height = event.height

    def on_closing(self):
        """When the app closes, close the database connection."""
        self.db.close()  # Close database connection properly
        self.destroy()  # Destroy the Tkinter window


if __name__ == "__main__":
    app = BodybuildingApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle window close
    app.mainloop()
