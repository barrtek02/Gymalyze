# live_detection.py

import time
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
        self.exercises_data = []  # List to store data for each exercise
        self.current_exercise = None
        self.current_repetition_count = 0
        self.current_feedback_display_list = []  # This will reset each frame for GUI display
        self.full_feedback_list = []  # This accumulates all feedback for saving to the database

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

        # Check if the exercise has changed
        if self.current_exercise != current_prediction:
            if self.current_exercise is not None:
                # Save the previous exercise data
                self.save_exercise_data()
            # Reset counters for the new exercise
            self.current_exercise = current_prediction
            self.current_repetition_count = repetition_count
            self.full_feedback_list = []  # Clear accumulated feedback for the new exercise
            self.current_exercise_start_time = time.time()  # Record start time
            self.last_feedback_time = 0  # Reset feedback timestamp
        else:
            # Update repetition count and feedback
            self.current_repetition_count = repetition_count
            if isinstance(current_feedback, list):
                current_time = time.time()
                if current_time - self.last_feedback_time >= 1:
                    # Append to full feedback list for database storage
                    self.full_feedback_list.extend(current_feedback)

                    # Update limited feedback display list
                    self.current_feedback_display_list = current_feedback
                    self.last_feedback_time = current_time
            else:
                print(f"Warning: current_feedback is not a list: {current_feedback}")

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
        if not self.current_feedback_display_list or not isinstance(
            self.current_feedback_display_list[0], tuple
        ):
            current_feedback_display = [
                ("Angle Correctness", ["No Feedback"]),
                ("Pose Correctness", ["Score: 0.0", "No Prediction"]),
            ]
        else:
            current_feedback_display = self.current_feedback_display_list

        # Format right-side feedback messages for readability with sections
        right_text = ""
        for section, messages in current_feedback_display:
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

    def format_feedback_display(self, feedback_list):
        """Format feedback list for display in the GUI."""
        if not feedback_list or not isinstance(feedback_list[0], tuple):
            feedback_list = [
                ("Angle Correctness", ["No Feedback"]),
                ("Pose Correctness", ["Score: 0.0", "No Prediction"]),
            ]

        right_text = ""
        for section, messages in feedback_list:
            right_text += f"{section}:\n"
            for msg in messages:
                right_text += f" - {msg}\n"
            right_text += "\n"

        return right_text

    def save_exercise_data(self):
        """Save data for the current exercise."""
        exercise = self.current_exercise
        repetitions = self.current_repetition_count

        # Calculate duration
        end_time = time.time()
        duration = (
            end_time - self.current_exercise_start_time
            if self.current_exercise_start_time
            else 0
        )

        # Extract feedback components
        angle_correctness, summary_pose_correctness_score, pose_correctness_scores, unique_grades = (
            self.extract_feedback_components(self.full_feedback_list)
        )

        # Insert into the exercise_sessions table and get the exercise_session_id
        session_id = self.controller.current_session_id
        exercise_session_id = self.controller.db.insert_exercise_session(
            session_id,
            exercise,
            repetitions,
            duration,
            len(angle_correctness),
            summary_pose_correctness_score,
            ", ".join(unique_grades) if unique_grades else "No Prediction",
        )

        if exercise_session_id is None:
            print("Failed to insert exercise session.")
            return

        # Insert each angle_correctness feedback into angle_correctness table
        for feedback in angle_correctness:
            if (
                    feedback.get("body_part")
                    and feedback.get("angle") is not None
                    and feedback.get("comment")
            ):
                time_of_appearance = self.calculate_time_of_appearance()
                self.controller.db.insert_angle_correctness(
                    exercise_session_id,
                    feedback.get("body_part"),
                    feedback.get("angle"),
                    feedback.get("comment"),
                    time_of_appearance=time_of_appearance
                )

        # Insert each individual pose_correctness score and grade
        for score, grade in zip(pose_correctness_scores, unique_grades):
            time_of_appearance = self.calculate_time_of_appearance()
            self.controller.db.insert_pose_correctness(
                exercise_session_id, score, grade, time_of_appearance=time_of_appearance
            )

        # Append to exercises data
        self.exercises_data.append(
            {
                "exercise": exercise,
                "repetitions": repetitions,
                "duration": duration,
                "angle_correctness_count": len(angle_correctness),
                "pose_correctness_score": summary_pose_correctness_score,
                "pose_correctness_grade": ", ".join(unique_grades) if unique_grades else "No Prediction",
            }
        )

        # Reset current exercise data
        self.current_exercise = None
        self.current_repetition_count = 0
        self.current_feedback_display_list = []  # Clear the display feedback list
        self.full_feedback_list = []  # Clear the full feedback list for the next exercise
        self.current_exercise_start_time = None

    def extract_feedback_components(self, feedback_list):
        """Extract feedback components from the feedback list."""
        angle_correctness = []
        pose_correctness_scores = []
        pose_correctness_grades = []

        print("Debug: Starting extract_feedback_components")
        print(f"Feedback List: {feedback_list}")

        for feedback in feedback_list:
            if isinstance(feedback, tuple) and len(feedback) == 2:
                section, messages = feedback
                if section == "Angle Correctness":
                    for msg in messages:
                        # Handle "No Feedback"
                        if msg.strip().lower() == "no feedback":
                            continue
                        # Assuming msg format: "BodyPart: Angle°\n(Comment)"
                        try:
                            body_part_part, rest = msg.split(":", 1)
                            angle_part, comment_part = rest.split("\n", 1)
                            angle_str = angle_part.strip().replace("°", "")
                            angle = float(angle_str)
                            comment = comment_part.strip("()")
                            angle_correctness.append(
                                {
                                    "body_part": body_part_part.strip(),
                                    "angle": angle,
                                    "comment": comment,
                                }
                            )
                        except ValueError:
                            print(
                                f"Warning: Unable to parse angle correctness message: {msg}"
                            )
                elif section == "Pose Correctness":
                    for msg in messages:
                        if msg.startswith("Score:"):
                            try:
                                score = float(msg.split("Score:")[1].strip())
                                pose_correctness_scores.append(score)
                            except ValueError:
                                print(
                                    f"Warning: Unable to parse score from message: {msg}"
                                )
                        else:
                            pose_correctness_grades.append(msg.strip())
            else:
                print(f"Warning: Unexpected feedback format: {feedback}")

        # Choose a specific score, e.g., the highest score (or however you want to summarize it)
        summary_pose_correctness_score = (
            max(pose_correctness_scores) if pose_correctness_scores else None
        )

        # Combine pose correctness grades into a single string or other desired structure
        unique_grades = list(set(pose_correctness_grades))

        print(f"Debug: Extracted angle_correctness: {angle_correctness}")
        print(f"Debug: Extracted summary_pose_correctness_score: {summary_pose_correctness_score}")
        print(f"Debug: Extracted unique grades: {unique_grades}")

        return angle_correctness, summary_pose_correctness_score, pose_correctness_scores, unique_grades

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
