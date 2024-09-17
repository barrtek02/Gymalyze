import tkinter as tk
from tkinter import PhotoImage, messagebox
from tkinter import ttk  # Import ttk for themed widgets
import sv_ttk  # Import Sun Valley theme

from gui.login_screen import LoginScreen
from gui.home_screen import HomeScreen
from gui.live_detection import LiveDetectionScreen
from gui.train_exercise import TrainSpecificScreen
from gui.upload_review import UploadVideoScreen
from gui.sign_up_screen import SignUpScreen
from utils.database import Database  # Import the database class
import os


class BodybuildingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gymalyze")
        self.geometry("400x400")
        self.set_app_icon()
        self.db = Database()

        # Container to hold the frames
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Dictionary to hold frames
        self.frames = {}
        self.create_frames()

        # Start by showing the login screen
        self.show_frame("LoginScreen")

        # Apply the Sun Valley Dark theme
        sv_ttk.set_theme("dark")  # Set the theme to dark

    def set_app_icon(self):
        """Set the app icon using the gymalyze icon from assets"""
        icon_path = os.path.join(
            os.path.dirname(__file__), "assets", "gymalyze_icon.png"
        )
        self.icon_img = PhotoImage(file=icon_path)  # Load the image
        self.iconphoto(False, self.icon_img)  # Set the image as the icon

    def create_frames(self):
        # Initialize all frames and pass the shared database instance
        for F in (
            LoginScreen,
            HomeScreen,
            LiveDetectionScreen,
            TrainSpecificScreen,
            UploadVideoScreen,
            SignUpScreen,
        ):
            frame = F(parent=self.container, controller=self, db=self.db)
            frame_name = F.__name__  # Get the name of the class as a string
            self.frames[frame_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, frame_name):
        """Bring the desired frame to the front using the frame name."""
        frame = self.frames[frame_name]
        frame.tkraise()  # Raise the selected frame to the front

    def on_closing(self):
        """When the app closes, close the database connection."""
        self.db.close()
        self.destroy()


if __name__ == "__main__":
    app = BodybuildingApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
