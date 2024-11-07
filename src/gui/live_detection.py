import threading
import tkinter as tk
from tkinter import Label
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.utils.pose_estimator import PoseEstimator
from src.utils.repetition_counter import RepetitionCounter
from src.utils.video_processor import VideoProcessor
from src.utils.imutils.video import WebcamVideoStream, FPS
from src.utils.exercise_evaluators import (
    SquatEvaluator,
    DeadliftEvaluator,
    BenchPressEvaluator,
    PushUpEvaluator,
    BicepCurlEvaluator,
)
from src.utils.sequence_comparator import SequenceComparator


class LiveDetectionScreen(tk.Frame):
    def __init__(self, parent: tk.Tk, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.current_similarity = None
        self.process_thread = None
        self.fps: FPS | None = None
        self.vs: WebcamVideoStream | None = None
        self.pose_estimator: PoseEstimator | None = None
        self.controller: tk.Tk = controller
        self.video_processor: VideoProcessor = VideoProcessor()
        self.repetition_counter = RepetitionCounter()
        self.sliding_window = []
        self.sliding_window_size = 50
        self.current_prediction = "No Prediction"  # Default value
        self.current_probability = 0.0
        self.current_landmarks = None
        self.frame_count = 0
        self.current_feedback = [
            ("Angle Correctness", ["No Feedback"]),
            ("Pose Correctness", ["Score: 0.0", "No Prediction"]),
        ]
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

        # Thread control variables
        self.stop_event = threading.Event()

        # Instantiate Evaluators
        self.evaluators = {
            "bench_press": BenchPressEvaluator(),
            "bicep_curl": BicepCurlEvaluator(),
            "squat": SquatEvaluator(),
            "deadlift": DeadliftEvaluator(),
            "push_up": PushUpEvaluator(),
        }

        # Load the dataset and labels
        dataset = np.load(
            r"C:\Users\barrt\PycharmProjects\Gymalyze\src\data\landmarks_data.npy",
            allow_pickle=True,
        )
        labels = np.load(
            r"C:\Users\barrt\PycharmProjects\Gymalyze\src\data\labels_data.npy",
            allow_pickle=True,
        )

        label_to_exercise = {
            0: "bench_press",
            1: "bicep_curl",
            2: "squat",
            3: "deadlift",
            4: "push_up",
        }

        self.sequence_comparator = SequenceComparator(
            dataset, labels, label_to_exercise
        )

    def on_show(self) -> None:
        """Start the webcam feed and update the frames."""
        self.stop_event.clear()  # Reset the stop event before starting the new thread

        self.current_prediction = "No Prediction"
        self.current_probability = 0.0
        self.sliding_window = []
        self.current_feedback = [
            ("Angle Correctness", ["No Feedback"]),
            ("Pose Correctness", ["Score: 0.0", "No Prediction"]),
        ]
        self.vs = WebcamVideoStream().start()
        self.fps = FPS().start()
        self.pose_estimator = PoseEstimator()
        self.repetition_counter = RepetitionCounter()

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
                        sequence = np.array(
                            self.video_processor.format_landmarks(self.sliding_window)
                        )
                        self.current_prediction, self.current_probability = (
                            self.classify_sliding_window(sequence)
                        )
                        self.repetition_counter.track_phases(
                            self.sliding_window, self.current_prediction
                        )
                        if self.frame_count % 30 == 0:

                            # Evaluate correctness if prediction is recognized
                            exercise_key = self.current_prediction.lower()
                            if exercise_key in self.evaluators:
                                evaluator = self.evaluators[exercise_key]
                                self.evaluator_feedback = evaluator.evaluate(
                                    pose_landmarks_raw.landmark
                                )
                            else:
                                self.evaluator_feedback = ["Good form!"]

                            average_similarity = self.sequence_comparator.compare(
                                sequence,
                                self.current_prediction.lower(),
                            )
                            self.current_similarity = average_similarity
                            self.generate_feedback()
                        self.sliding_window.pop(0)
                self.frame_count += 1

    def generate_feedback(self):
        """
        Generates feedback based on the similarity score and updates self.current_feedback.
        """
        threshold_excellent = 0.9
        threshold_very_good = 0.85
        threshold_good = 0.8
        threshold_wrong = 0.7

        # Clear existing feedback
        self.current_feedback = []

        # Angle correctness feedbacks
        angle_feedback = (
            self.evaluator_feedback if hasattr(self, "evaluator_feedback") else []
        )

        # Pose correctness feedbacks
        if self.current_similarity is not None:
            score = f"Score: {round(self.current_similarity, 2)}"
        else:
            score = "Score: N/A"

        if self.current_similarity is not None:
            if self.current_similarity >= threshold_excellent:
                qualitative_feedback = "Perfect"
            elif self.current_similarity >= threshold_very_good:
                qualitative_feedback = "Very Good"
            elif self.current_similarity >= threshold_good:
                qualitative_feedback = "Good"
            elif self.current_similarity >= threshold_wrong:
                qualitative_feedback = "Wrong"
            else:
                qualitative_feedback = "Horrible"
        else:
            qualitative_feedback = "No Prediction"

        pose_feedback = [score, qualitative_feedback]

        # Set feedback sections
        self.current_feedback.append(("Angle Correctness", angle_feedback))
        self.current_feedback.append(("Pose Correctness", pose_feedback))

    def update_frame(self) -> None:
        """Update the video feed frame."""
        if self.vs is None:
            return

        frame = self.vs.read()
        self.pose_estimator.draw_pose(frame, self.current_landmarks)
        if frame is None or self.fps is None:
            return

        self.fps.update()

        # Collect left side information
        # Format prediction text with desired layout
        prediction_name = self.current_prediction.replace("_", " ").title()
        prediction_text = (
            f"FPS: {int(self.fps.fps())}\n\n"
            f"Prediction:\n"
            f"{prediction_name} ({round(self.current_probability, 2)}%)\n\n"
            f"Repetitions:\n"
            f"{self.repetition_counter.get_repetition_count(self.current_prediction)}"
        )

        # Update left label with formatted text
        self.left_label.config(text=prediction_text, font=("Helvetica", 18))

        # Ensure self.current_feedback is in the correct format
        if not self.current_feedback or not isinstance(self.current_feedback[0], tuple):
            self.current_feedback = [
                ("Angle Correctness", ["No Feedback"]),
                ("Pose Correctness", ["Score: 0.0", "No Prediction"]),
            ]

        # Format right-side feedback messages for readability with sections
        right_text = ""
        for section, messages in self.current_feedback:
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
        frame_resized = cv2.resize(frame_rgb, (desired_width, desired_height))
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = img_tk
        self.video_label.config(image=img_tk)

        # Schedule the next update
        self.after(10, self.update_frame)

    def classify_sliding_window(self, sequence: np.ndarray) -> tuple[str, np.ndarray]:
        """Classify the current sliding window."""

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

        if self.repetition_counter is not None:
            self.repetition_counter = None

        self.controller.show_frame("HomeScreen")
