import tkinter as tk
from datetime import datetime
from tkinter import PhotoImage
from tkinter import ttk
from tkinter import messagebox
import sv_ttk
import torch

from gui.home_screen import HomeScreen
from gui.live_detection import LiveDetectionScreen
from gui.train_exercise import TrainSpecificScreen
from gui.upload_review import UploadVideoScreen
from src.gui.exercise_library_screen import ExerciseLibraryScreen
from src.models.lstm import ExerciseLSTM
from src.models.pose_autoencoder import PoseAutoencoder
from utils.database import Database
import os

CLASSIFICATION_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "saved_models/lstm_v3.pth"
)
AUTOENCODER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "saved_models/autoencoder_v1.pth"
)


class BodybuildingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.session_start_time = None
        self.device: torch.device | None = None
        self.icon_img: PhotoImage | None = None
        self.classification_model: ExerciseLSTM | None = None
        self.autoencoder: PoseAutoencoder | None = None

        self.set_window_geometry()
        self.set_app_icon()
        self.db = Database()
        self.current_session_id = None  # To keep track of the current session

        # Container for holding frames (screens)
        self.container: ttk.Frame = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Dictionary to hold all screens (frames)
        self.frames: dict[str, ttk.Frame] = {}
        self.create_frames()

        # Start by showing the home screen
        self.show_frame("HomeScreen")

        sv_ttk.set_theme("dark")
        self.bind("<Configure>", self.on_resize)

        self.load_model()
        self.load_autoencoder()

    def set_window_geometry(self) -> None:
        """Set the initial window geometry and handle resizing."""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.5)
        window_height = int(screen_height * 0.5)
        x_position = int((screen_width - window_width) / 2)
        y_position = int((screen_height - window_height) / 2)
        self.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        self.minsize(400, 300)  # Minimum window size of 400x300

    def set_app_icon(self) -> None:
        """Set the app icon using the gymalyze icon from assets."""
        icon_path = os.path.join(
            os.path.dirname(__file__), "assets", "gymalyze_icon.png"
        )
        if os.path.exists(icon_path):
            self.icon_img = PhotoImage(file=icon_path)
            self.iconphoto(False, self.icon_img)

    def create_frames(self) -> None:
        """Initialize and store all frames in a dictionary for easy access."""
        for F in (
            HomeScreen,
            LiveDetectionScreen,
            TrainSpecificScreen,
            UploadVideoScreen,
            ExerciseLibraryScreen,
        ):
            frame = F(parent=self.container, controller=self)
            frame_name = F.__name__
            self.frames[frame_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

    def load_model(
        self,
        input_size: int = 33 * 4,
        hidden_size: int = 512,
        num_layers: int = 1,
        num_classes: int = 5,
    ) -> None:
        """Loads the model in a background thread."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.classification_model = ExerciseLSTM(
                input_size, hidden_size, num_layers, num_classes
            )

            self.classification_model.load_state_dict(
                torch.load(
                    CLASSIFICATION_MODEL_PATH,
                    map_location=self.device,
                )
            )
            self.classification_model.to(self.device)
            self.classification_model.eval()

        except Exception as e:
            messagebox.showerror("Model Load Error", f"Error loading model: {str(e)}")

    def load_autoencoder(
        self, input_dim: int = 33 * 4, hidden_dim: int = 1024, latent_dim: int = 128
    ) -> None:
        """Load the pose autoencoder model."""
        try:
            self.autoencoder = PoseAutoencoder(input_dim, hidden_dim, latent_dim)
            self.autoencoder.load_state_dict(
                torch.load(
                    AUTOENCODER_MODEL_PATH,
                    map_location=self.device,
                )
            )
            self.autoencoder.to(self.device)
            self.autoencoder.eval()

        except Exception as e:
            messagebox.showerror("Model Load Error", f"Error loading model: {str(e)}")

    def show_frame(self, frame_name) -> None:
        """Show the desired frame by bringing it to the front."""
        frame = self.frames[frame_name]
        frame.tkraise()  # Raise the selected frame to the front
        if frame_name == "LiveDetectionScreen":
            if self.classification_model is None:
                messagebox.showwarning(
                    "Model Not Ready", "Model is still loading. Please wait."
                )
            else:
                frame.on_show()

    def on_resize(self, event) -> None:
        """Optional: Handle resizing logic."""

    def start_new_session(self):
        """Start a new session and store the session ID."""
        self.session_start_time = datetime.now()  # Record the session start time
        self.current_session_id = self.db.insert_session(self.session_start_time, None)

    def end_current_session(self):
        """End the current session by calculating duration and updating the database."""
        if self.session_start_time:
            end_time = datetime.now()
            duration = (end_time - self.session_start_time).total_seconds()
            self.db.update_session_duration(self.current_session_id, duration)
            self.session_start_time = None  # Reset session start time

    def on_closing(self) -> None:
        """When the app closes, close the database connection."""
        self.db.close()
        self.destroy()  # Destroy the Tkinter window


if __name__ == "__main__":
    app = BodybuildingApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle window close
    app.mainloop()
