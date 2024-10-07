import tkinter as tk
from tkinter import filedialog, Text, Button
from typing import Any

import cv2
import numpy as np
from numpy import signedinteger, ndarray, dtype
from numpy.core import long

from src.utils.video_processor import VideoProcessor
from src.utils.pose_estimator import PoseEstimator
from datetime import timedelta


class UploadVideoScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.video_processor = VideoProcessor()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Text area to display results
        self.result_text = Text(self, height=20, width=80)
        self.result_text.grid(row=1, column=0, padx=10, pady=10)

        # Button to upload video
        upload_button = Button(
            self, text="Upload Video", command=self.upload_video, width=30, height=2
        )
        upload_button.grid(row=0, column=0, padx=10, pady=10)

    def upload_video(self) -> None:
        """Open file dialog for user to select a video."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )

        if file_path:
            self.process_video(file_path)

    def process_video(self, video_path: str) -> None:
        """Process the uploaded video for exercise classification with timestamps."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        sliding_window = []
        sliding_window_size = 50
        analysis_results = []

        # Initialize variables for time tracking
        current_exercise = None
        start_time = 0
        confidences = []

        pose_estimator = PoseEstimator()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame (for example, every 4th frame)
            if frame_count % 4 == 0:

                pose_landmarks_raw = pose_estimator.estimate_pose(frame)
                if pose_landmarks_raw:
                    pose_landmarks = self.video_processor.process_pose_landmarks(
                        pose_landmarks_raw
                    )
                    sliding_window.append(pose_landmarks)

                    # If sliding window is full, classify the sequence
                    if len(sliding_window) == sliding_window_size:
                        sequence = np.array(
                            self.video_processor.format_landmarks(sliding_window)
                        )
                        exercise_class, confidence = self.classify_sequence(sequence)
                        confidences.append(confidence)

                        # Get the current time in the video
                        current_time = frame_count / fps

                        # If the exercise has changed, save the previous exercise with timestamps
                        if (
                            current_exercise is not None
                            and current_exercise != exercise_class
                        ):
                            end_time = current_time
                            avg_confidence = np.mean(confidences)
                            self.add_analysis_result(
                                current_exercise,
                                start_time,
                                end_time,
                                avg_confidence,
                                analysis_results,
                            )
                            confidences = []  # Reset confidence list
                            start_time = current_time

                        # Update current exercise
                        current_exercise = exercise_class
                        sliding_window.pop(0)

            frame_count += 1

        # Handle the last exercise in the video
        if current_exercise is not None:
            end_time = frame_count / fps
            avg_confidence = np.mean(confidences)
            self.add_analysis_result(
                current_exercise, start_time, end_time, avg_confidence, analysis_results
            )

        cap.release()

        # Display results in the text area
        self.display_analysis(analysis_results)

    def classify_sequence(
        self, sequence: np.ndarray
    ) -> tuple[signedinteger[Any] | long, ndarray[Any, dtype[signedinteger[Any]]]]:
        """Classify a sequence of pose landmarks and return the predicted exercise and confidence."""
        if self.controller.classification_model is None:
            raise ValueError("Model is None, ensure the model is loaded correctly!")
        probabilities = self.video_processor.classify_sequence(
            self.controller.classification_model, sequence, self.controller.device
        )
        max_prob_index = np.argmax(probabilities)
        confidence = probabilities[max_prob_index] * 100
        return max_prob_index, confidence  # Return class index and confidence

    def add_analysis_result(
        self,
        exercise_class: int,
        start_time: float,
        end_time: float,
        avg_confidence: float,
        analysis_results: list[dict[str, Any]],
    ) -> None:
        """Add the classified exercise with timestamps and average confidence to the results list."""
        result = {
            "exercise": f"Class {self.video_processor.convert_index_to_exercise_name(exercise_class)}",  # You can map the class index to actual exercise names if available
            "start_time": timedelta(seconds=int(start_time)),
            "end_time": timedelta(seconds=int(end_time)),
            "confidence": round(avg_confidence, 2),
        }
        analysis_results.append(result)

    def display_analysis(self, analysis_results: list[dict[str, Any]]) -> None:
        """Display the analysis results in the text area."""
        self.result_text.delete(1.0, tk.END)  # Clear previous results
        for result in analysis_results:
            display_text = (
                f"Exercise: {result['exercise']}\n"
                f"Start Time: {result['start_time']}\n"
                f"End Time: {result['end_time']}\n"
                f"Average Confidence: {result['confidence']}%\n\n"
            )
            self.result_text.insert(tk.END, display_text)
