import tkinter as tk
from tkinter import Label
import cv2
import torch
from PIL import Image, ImageTk

from src.models.lstm import ExerciseLSTM
from src.utils.database import Database
from src.utils.video_processor import VideoProcessor
from src.utils.imutils.video import WebcamVideoStream, FPS


class LiveDetectionScreen(tk.Frame):
    def __init__(
        self,
        parent: tk.Tk,
        controller: tk.Tk,
        db: Database,
        model: ExerciseLSTM,
        device: torch.device,
    ) -> None:
        super().__init__(parent)
        self.fps: FPS | None = None
        self.vs: WebcamVideoStream | None = None
        self.controller: tk.Tk = controller
        self.db: Database = db
        self.model: ExerciseLSTM = model
        self.device: torch.device = device
        self.video_processor: VideoProcessor = VideoProcessor()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # Create a label for the video feed
        self.video_label = Label(self)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        exit_button = tk.Button(
            self,
            text="Exit Detection",
            command=self.on_closing,
            width=30,
            height=2,
        )
        exit_button.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        # Start updating frames

    def on_show(self) -> None:
        self.vs = WebcamVideoStream().start()  # Start webcam capture
        self.fps = FPS().start()  # Start FPS counter
        self.update_frame()

    def update_frame(self) -> None:
        if self.vs is None:
            return

        frame = self.vs.read()
        if frame is None:
            return

        if self.fps is None:
            return
        # Update FPS counter
        self.fps.update()

        # Add the FPS text to the frame
        fps_text = f"FPS: {int(self.fps.fps())}"
        cv2.putText(
            frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the image to PIL format
        img = Image.fromarray(frame_rgb)

        # Convert the PIL image to ImageTk format
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the video label with the new frame
        self.video_label.imgtk = img_tk
        self.video_label.config(image=img_tk)

        # Call update_frame again after 10 ms
        self.after(10, self.update_frame)

    def on_closing(self) -> None:
        self.vs.stop()
        self.vs = None

        self.fps.stop()
        self.fps = None

        self.controller.show_frame("HomeScreen")
