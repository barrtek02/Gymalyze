import tkinter as tk
from tkinter import Label

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.utils.frame_processor import FrameProcessor


class LiveDetectionScreen(tk.Frame):
    def __init__(self, parent: tk.Tk, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.controller: tk.Tk = controller

        # Initialize the FrameProcessor
        self.frame_processor = FrameProcessor(controller)

        # Configure grid layout with a 3x3 grid
        for i in range(3):
            self.grid_columnconfigure(i, weight=1)
            self.grid_rowconfigure(i, weight=1)

        # Center cell (1,1) for the video feed
        self.video_label = Label(self)
        self.video_label.grid(row=1, column=1, padx=10, pady=10, sticky="n")

        # Left label for displaying numeric information in cell (1,0)
        self.left_label = Label(
            self,
            text="",
            justify="center",
            font=("Helvetica", 18),
            width=30,
            wraplength=250,
            anchor="center",
        )
        self.left_label.grid(row=1, column=0, padx=10, pady=10, sticky="n")

        # Right label for displaying feedback in cell (1,2)
        self.right_label = Label(
            self,
            text="",
            justify="center",
            font=("Helvetica", 18),
            width=30,
            wraplength=250,
            anchor="center",
        )
        self.right_label.grid(row=1, column=2, padx=10, pady=10, sticky="n")

        # Bottom-center cell (2,1) for the exit button
        exit_button = tk.Button(
            self,
            text="Exit Detection",
            command=self.on_closing,
            width=30,
            height=2,
        )
        exit_button.grid(row=2, column=1, padx=20, pady=20, sticky="nsew")

    def on_show(self) -> None:
        """Start the webcam feed and update the frames."""
        self.frame_processor.start()
        # Start the regular UI frame update
        self.update_frame()

    def update_frame(self) -> None:
        """Update the video feed frame."""
        frame = self.frame_processor.get_current_frame()
        if frame is None:
            self.after(10, self.update_frame)
            return

        # Get data from the frame processor
        data = self.frame_processor.get_current_data()
        fps = data["fps"]
        current_prediction = data["current_prediction"]
        current_probability = data["current_probability"]
        repetition_count = data["repetition_count"]
        current_feedback = data["current_feedback"]

        # Collect left side information
        # Format prediction text with desired layout
        prediction_name = current_prediction.replace("_", " ").title()
        prediction_text = (
            f"FPS: {fps}\n\n"
            f"Prediction:\n"
            f"{prediction_name} ({round(current_probability, 2)}%)\n\n"
            f"Repetitions:\n"
            f"{repetition_count}"
        )

        # Update left label with formatted text
        self.left_label.config(text=prediction_text, font=("Helvetica", 18))

        # Ensure current_feedback is in the correct format
        if not current_feedback or not isinstance(current_feedback[0], tuple):
            current_feedback = [
                ("Angle Correctness", ["No Feedback"]),
                ("Pose Correctness", ["Score: 0.0", "No Prediction"]),
            ]

        # Format right-side feedback messages for readability with sections
        right_text = ""
        for section, messages in current_feedback:
            right_text += f"{section}:\n"
            for msg in messages:
                right_text += f" - {msg}\n"
            right_text += "\n"  # Add space between sections

        # Update right label with feedback text
        self.right_label.config(text=right_text.strip(), font=("Helvetica", 18))

        # Convert frame to RGB and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to make the camera view larger
        desired_width = 800  # Set your desired width
        desired_height = 600  # Set your desired height
        # frame_resized = cv2.resize(frame_rgb, (desired_width, desired_height))
        frame_resized = black_image = np.zeros(
            (desired_height, desired_width, 3), dtype=np.uint8
        )
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = img_tk
        self.video_label.config(image=img_tk)

        # Schedule the next update
        self.after(10, self.update_frame)

    def on_closing(self) -> None:
        """Stop the video stream and processing thread when closing."""
        self.frame_processor.stop()
        self.controller.show_frame("HomeScreen")
