import threading
import tkinter as tk
from tkinter import Label
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from numpy import ndarray, dtype, signedinteger

from src.models.lstm import ExerciseLSTM
from src.utils.database import Database
from src.utils.pose_estimator import PoseEstimator
from src.utils.video_processor import VideoProcessor
from src.utils.imutils.video import WebcamVideoStream, FPS


class LiveDetectionScreen(tk.Frame):
    def __init__(self, parent: tk.Tk, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.process_thread = None
        self.fps: FPS | None = None
        self.vs: WebcamVideoStream | None = None
        self.pose_estimator: PoseEstimator | None = None
        self.controller: tk.Tk = controller
        self.video_processor: VideoProcessor = VideoProcessor()
        self.sliding_window = []
        self.sliding_window_size = 50
        self.current_prediction = "No Prediction"  # Default value
        self.current_probability = 0.0
        self.current_landmarks = None
        self.frame_count = 0

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

        # Thread control variables
        self.stop_event = threading.Event()

    def on_show(self) -> None:
        """Start the webcam feed and update the frames."""
        self.stop_event.clear()  # Reset the stop event before starting the new thread

        self.current_prediction = "No Prediction"  # Default value
        self.current_probability = 0.0
        self.sliding_window = []

        self.vs = WebcamVideoStream().start()
        self.fps = FPS().start()
        self.pose_estimator = PoseEstimator()

        # Start the thread that processes frames
        self.process_thread = threading.Thread(target=self.process_frames)
        self.process_thread.start()

        # Start the regular UI frame update
        self.update_frame()

    def process_frames(self) -> None:
        """Thread function to process video frames in the background."""

        while not self.stop_event.is_set():
            if self.vs:
                frame = self.vs.read()
                if frame is None:
                    continue

                pose_landmarks_raw = self.pose_estimator.estimate_pose(frame)
                self.current_landmarks = pose_landmarks_raw
                if pose_landmarks_raw:
                    self.sliding_window.append(
                        self.video_processor.process_pose_landmarks(pose_landmarks_raw)
                    )

                    if len(self.sliding_window) == self.sliding_window_size:
                        self.current_prediction, self.current_probability = (
                            self.classify_sliding_window()
                        )
                        self.sliding_window.pop(0)

    def update_frame(self) -> None:
        """Update the video feed frame."""
        if self.vs is None:
            return

        frame = self.vs.read()
        self.pose_estimator.draw_pose(frame, self.current_landmarks)
        if frame is None:
            return

        if self.fps is None:
            return

        # Update FPS counter
        self.fps.update()

        # Add the FPS text
        fps_text = f"FPS: {int(self.fps.fps())}"
        cv2.putText(
            frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # Add the prediction text
        prediction_text = f"Prediction: {self.current_prediction} {round(self.current_probability, 2)}%"
        cv2.putText(
            frame,
            prediction_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
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

    def classify_sliding_window(
        self,
    ) -> tuple[str, ndarray[Any, dtype[signedinteger[Any]]]]:
        """Classify the current sliding window."""
        sequence = np.array(self.video_processor.format_landmarks(self.sliding_window))

        if self.controller.classification_model is None:
            raise ValueError("Model is None, ensure the model is loaded correctly!")

        probabilities = self.video_processor.classify_sequence(
            self.controller.classification_model, sequence, self.controller.device
        )
        max_prob_index = np.argmax(probabilities)

        return (
            self.video_processor.convert_index_to_exercise_name(max_prob_index),
            probabilities[max_prob_index] * 100,
        )

    def on_closing(self) -> None:
        """Stop the video stream and processing thread when closing."""
        self.stop_event.set()  # Signal the thread to stop

        if self.process_thread is not None:
            self.process_thread.join()  # Ensure the thread is properly closed
            self.process_thread = None

        if self.vs is not None:
            self.vs.stop()  # Stop the video stream
            self.vs = None

        if self.fps is not None:
            self.fps.stop()  # Stop the FPS counter
            self.fps = None

        if self.pose_estimator is not None:
            self.pose_estimator = None

        self.controller.show_frame("HomeScreen")
