# live_detection.py
import logging
import time
import tkinter as tk
from tkinter import Label, ttk

import cv2
from PIL import Image, ImageTk

from utils.frame_processor import FrameProcessor


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
        exit_button = ttk.Button(
            self,
            text="Exit Detection",
            command=self.on_closing,
            width=30,
        )
        exit_button.grid(row=2, column=1, padx=20, pady=20, sticky="nsew")
        self.exercises_data = []  # List to store data for each exercise
        self.current_exercise = None
        self.current_repetition_count = 0
        self.current_feedback_display_list = (
            []
        )  # This will reset each frame for GUI display
        self.full_feedback_list = (
            []
        )  # This accumulates all feedback for saving to the database

        self.current_exercise_start_time = (
            None  # Track the start time of the current exercise
        )
        self.last_feedback_time = 0  # Timestamp of the last feedback generation

    def on_show(self) -> None:
        """Start the webcam feed and update the frames."""
        # Start a new session
        self.controller.start_new_session()
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

        if "score" not in current_feedback or current_feedback["score"] is None:
            current_feedback["score"] = 0.0  # Default score if not provided
        # Check if the exercise has changed
        if self.current_exercise != current_prediction:
            if self.current_exercise is not None:
                # Save the previous exercise data
                self.save_exercise_data()

            # Reset counters for the new exercise
            self.current_exercise = current_prediction
            self.current_repetition_count = repetition_count
            self.full_feedback_list = (
                []
            )  # Clear accumulated feedback for the new exercise
            self.current_exercise_start_time = time.time()  # Record start time
            self.last_feedback_time = 0  # Reset feedback timestamp
        else:
            # Update repetition count and feedback
            self.current_repetition_count = repetition_count
            current_time = time.time()
            if (
                isinstance(current_feedback, dict)
                and current_time - self.last_feedback_time >= 1
            ):
                # Append feedback details, including score, to the full feedback list
                feedback_with_score = current_feedback.copy()
                if "details" not in feedback_with_score:
                    feedback_with_score["details"] = []
                self.full_feedback_list.append(feedback_with_score)

                # Update the current feedback display list for UI
                self.current_feedback_display_list = current_feedback.get("details", [])
                self.last_feedback_time = current_time

        # Extract exercise score and feedback details
        exercise_score = current_feedback.get("score", None)
        feedback_details = current_feedback.get("details", [])

        # Format exercise score
        score_text = (
            f"Exercise Score: {exercise_score:.2f}%"
            if exercise_score is not None
            else "No Score Available"
        )

        # Format right-side feedback text
        right_text = f"{score_text}\n\n"

        for feedback in feedback_details:
            # Extract details from feedback
            feedback.get("Frame", "N/A")
            friendly_angle = feedback.get("Friendly Angle", "Unknown Angle")
            input_angle = feedback.get("Input Angle (degrees)", "N/A")
            reconstructed_angle = feedback.get("Reconstructed Angle (degrees)", "N/A")
            feedback.get("Deviation (degrees)", "N/A")
            threshold = feedback.get("Threshold (degrees)", "N/A")
            feedback_message = feedback.get("Feedback", "No Feedback")

            # Format angles and threshold
            input_angle_str = (
                f"{input_angle:.2f}°"
                if isinstance(input_angle, (int, float))
                else input_angle
            )
            reconstructed_angle_str = (
                f"{reconstructed_angle:.2f}°"
                if isinstance(reconstructed_angle, (int, float))
                else reconstructed_angle
            )
            threshold_str = (
                f"± {threshold:.2f}°"
                if isinstance(threshold, (int, float))
                else threshold
            )

            # Add formatted text to `right_text`
            right_text += f"{friendly_angle}:\n"
            right_text += f"Input Angle: {input_angle_str}\n"
            right_text += f"Expected Angle: {reconstructed_angle_str} {threshold_str}\n"
            right_text += f"Feedback: {feedback_message}\n\n"

        # Update right label with the formatted text
        self.right_label.config(text=right_text.strip(), font=("Helvetica", 18))

        # Update the left label with prediction and repetition info
        prediction_name = current_prediction.replace("_", " ").title()
        prediction_text = (
            f"FPS: {fps}\n\n"
            f"Prediction:\n"
            f"{prediction_name} ({round(current_probability, 2)}%)\n\n"
            f"Repetitions:\n"
            f"{repetition_count}"
        )
        self.left_label.config(text=prediction_text, font=("Helvetica", 18))

        # Convert frame to RGB and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        desired_width = 800  # Set your desired width
        desired_height = 600  # Set your desired height
        frame_resized = cv2.resize(frame_rgb, (desired_width, desired_height))
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = img_tk
        self.video_label.config(image=img_tk)

        # Schedule the next update
        self.after(10, self.update_frame)

    def format_feedback_display(self, feedback_list):
        """Format feedback for display in the GUI."""
        unique_feedback = set()  # Track unique feedback
        right_text = ""

        # Display the exercise score
        if (
            hasattr(self.frame_processor, "current_score")
            and self.frame_processor.current_score is not None
        ):
            right_text += (
                f"Exercise Score: {self.frame_processor.current_score:.2f}%\n\n"
            )

        for feedback in feedback_list:
            if isinstance(feedback, dict):  # Ensure feedback is properly structured
                # Extract feedback details
                angle = feedback.get("Angle", "Unknown Angle")
                input_angle = feedback.get("Input Angle (degrees)", "N/A")
                reconstructed_angle = feedback.get(
                    "Reconstructed Angle (degrees)", "N/A"
                )
                threshold = feedback.get("Threshold (degrees)", "N/A")
                feedback_message = feedback.get("Feedback", "No Feedback")

                # Avoid duplicate feedback messages
                if feedback_message not in unique_feedback:
                    unique_feedback.add(feedback_message)
                    # Format the feedback for display
                    right_text += (
                        f"Angle: {angle}\n"
                        f"Input Angle: {input_angle:.1f}°\n"
                        f"Reconstructed Angle: {reconstructed_angle:.1f}°\n"
                        f"Threshold: {threshold:.1f}°\n"
                        f"Feedback: {feedback_message}\n\n"
                    )

        return right_text.strip() if right_text else "No Feedback Available"

    def save_exercise_data(self):
        """Save data for the current exercise."""
        exercise = self.current_exercise
        repetitions = self.current_repetition_count

        # Calculate duration
        end_time = time.time()
        duration = (
            round(end_time - self.current_exercise_start_time, 2)
            if self.current_exercise_start_time
            else 0.0
        )

        # Extract feedback components
        angle_correctness, pose_correctness_scores = self.extract_feedback_components(
            self.full_feedback_list
        )

        # Calculate the summary pose correctness score (e.g., average or max)
        summary_pose_correctness_score = (
            round(sum(pose_correctness_scores) / len(pose_correctness_scores), 2)
            if pose_correctness_scores
            else None
        )

        # Insert exercise session into the database
        session_id = self.controller.current_session_id
        exercise_session_id = self.controller.db.insert_exercise_session(
            session_id,
            exercise,
            repetitions,
            duration,
            len(angle_correctness),
            summary_pose_correctness_score,
        )

        if exercise_session_id is None:
            logging.error("Failed to insert exercise session.")
            return

        # Save individual angle correctness feedback
        for feedback in angle_correctness:
            angle = round(feedback["angle"], 2)
            expected_angle = round(feedback["expected_angle"], 2)
            threshold = round(feedback["threshold"], 2)

            self.controller.db.insert_angle_correctness(
                exercise_session_id,
                feedback["angle_name"],
                angle,
                expected_angle,
                threshold,
                feedback["comment"],
                self.calculate_time_of_appearance(),
            )

        # Save individual pose correctness scores
        for score in pose_correctness_scores:
            self.controller.db.insert_pose_correctness(
                exercise_session_id,
                round(score, 2),
                self.calculate_time_of_appearance(),
            )

        # Reset exercise data
        self.current_exercise = None
        self.current_repetition_count = 0
        self.current_feedback_display_list = []
        self.full_feedback_list = []
        self.current_exercise_start_time = None

    def extract_feedback_components(self, feedback_list):
        """Extract feedback components for saving."""
        angle_correctness = []
        pose_correctness_scores = []

        for feedback in feedback_list:
            if isinstance(feedback, dict):
                # Extract angle correctness data
                if "details" in feedback:
                    for detail in feedback["details"]:
                        angle_correctness.append(
                            {
                                "angle_name": detail.get(
                                    "Friendly Angle", "Unknown Angle"
                                ),
                                "angle": detail.get("Input Angle (degrees)", 0.0),
                                "expected_angle": detail.get(
                                    "Reconstructed Angle (degrees)", 0.0
                                ),
                                "threshold": detail.get("Threshold (degrees)", 0.0),
                                "comment": detail.get("Feedback", "No Feedback"),
                            }
                        )
                # Extract pose correctness score
                if "score" in feedback:
                    pose_correctness_scores.append(feedback["score"])

        return angle_correctness, pose_correctness_scores

    def calculate_time_of_appearance(self):
        """Calculate the time of appearance in seconds since the start of the session."""
        # Assuming session start time is tracked when the session begins
        if self.current_exercise_start_time is not None:
            return round(time.time() - self.current_exercise_start_time, 2)
        return 0  # Default to 0 if there's no start time set

    def on_closing(self) -> None:
        """Stop the video stream and processing thread when closing."""
        self.frame_processor.stop()
        # Save the last exercise data if any
        if self.current_exercise is not None:
            self.save_exercise_data()
        # Save all exercises data to the database (already handled in save_exercise_data)
        # End the current session
        self.controller.end_current_session()
        self.controller.show_frame("HomeScreen")
