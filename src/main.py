import tkinter as tk
from tkinter import PhotoImage
from tkinter import ttk
from tkinter import messagebox
import sv_ttk
import torch

from gui.login_screen import LoginScreen
from gui.home_screen import HomeScreen
from gui.live_detection import LiveDetectionScreen
from gui.train_exercise import TrainSpecificScreen
from gui.upload_review import UploadVideoScreen
from gui.sign_up_screen import SignUpScreen
from src.models.lstm import ExerciseLSTM
from utils.database import Database
import os


class BodybuildingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.device: torch.device | None = None
        self.icon_img: PhotoImage | None = None
        self.classification_model: ExerciseLSTM | None = None

        self.set_window_geometry()
        self.set_app_icon()
        self.db = Database()

        # Container for holding frames (screens)
        self.container: ttk.Frame = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Dictionary to hold all screens (frames)
        self.frames: dict[str, ttk.Frame] = {}
        self.create_frames()

        # Start by showing the login screen
        self.show_frame("LoginScreen")

        sv_ttk.set_theme("dark")
        self.bind("<Configure>", self.on_resize)

        self.load_model()

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
            LoginScreen,
            HomeScreen,
            LiveDetectionScreen,
            TrainSpecificScreen,
            UploadVideoScreen,
            SignUpScreen,
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
        hidden_size: int = 128,
        num_layers: int = 2,
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
                    r"C:\Users\barrt\PycharmProjects\Gymalyze\src\saved_models\lstm_v1.pth",
                    map_location=self.device,
                )
            )
            self.classification_model.to(self.device)
            self.classification_model.eval()

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
        width = event.width
        height = event.height

    def on_closing(self) -> None:
        """When the app closes, close the database connection."""
        self.db.close()
        self.destroy()  # Destroy the Tkinter window


if __name__ == "__main__":
    app = BodybuildingApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle window close
    app.mainloop()
