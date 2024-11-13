# upload_review.py
import tkinter as tk
from tkinter import filedialog, Text, Button
from src.utils.frame_processor import FrameProcessor
from datetime import timedelta, datetime


class UploadVideoScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.frame_processor = FrameProcessor(controller)
        self.exercises_data = []  # To store data for each exercise

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # Initialize session details
        self.session_start_time = None
        self.session_duration = None
        # Text area to display results
        self.result_text = Text(self, height=30, width=100)
        self.result_text.grid(row=1, column=0, padx=10, pady=10)

        # Button to upload video
        upload_button = Button(
            self, text="Upload Video", command=self.upload_video, width=30, height=2
        )
        upload_button.grid(row=0, column=0, padx=10, pady=10)

        # Return button to go back to the home screen
        return_button = Button(
            self, text="Return", command=lambda: controller.show_frame("HomeScreen"), width=30, height=2
        )
        return_button.grid(row=2, column=0, padx=10, pady=10, sticky="s")

    def upload_video(self) -> None:
        """Open file dialog for user to select a video."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )

        if file_path:
            # Start session time tracking
            self.session_start_time = datetime.now()

            # Process the video
            analysis_results, feedback_results = (
                self.frame_processor.process_uploaded_video(file_path)
            )

            # Calculate session duration
            self.session_duration = sum(
                (result["end_time"] - result["start_time"]).total_seconds()
                for result in analysis_results
            )

            # Save session data to database
            self.save_session_data()

            # Save exercise data in the same format as live detection
            self.save_analysis_data(analysis_results, feedback_results)
            self.display_analysis(analysis_results, feedback_results)

    def save_session_data(self):
        """Save session data to the database."""
        if self.session_start_time and self.session_duration is not None:
            # Insert session data with start time as date and calculated duration
            self.controller.current_session_id = self.controller.db.insert_session(
                self.session_start_time, self.session_duration
            )

    from datetime import datetime

    from datetime import datetime, timedelta

    def save_analysis_data(self, analysis_results, feedback_results):
        """Save analysis data in the same format as live_detection."""
        for analysis in analysis_results:
            # Parse the exercise data and remove "Class " prefix
            exercise = analysis["exercise"].replace("Class ", "")
            start_time = analysis["start_time"]
            end_time = analysis["end_time"]
            duration = (end_time - start_time).total_seconds()
            confidence = analysis["confidence"]

            # Set repetition count: 0 if no meaningful prediction
            repetitions = 0 if exercise == "No Prediction" else 1

            # Get relevant feedback for this exercise
            feedback_for_exercise = [
                fb for fb in feedback_results if fb["exercise"] == exercise
            ]

            # Extract feedback components
            angle_correctness, pose_correctness_scores, pose_correctness_grades = (
                self.extract_feedback_components(feedback_for_exercise)
            )

            # Adjust the start_time to calculate time_of_appearance correctly
            # Check if start_time is a timedelta and convert it to seconds if needed
            if isinstance(start_time, timedelta):
                time_of_appearance = start_time.total_seconds()
            else:
                time_of_appearance = (start_time - self.session_start_time).total_seconds()

            # Save each exercise session
            session_id = self.controller.current_session_id
            exercise_session_id = self.controller.db.insert_exercise_session(
                session_id,
                exercise,
                repetitions,
                duration,
                len(angle_correctness),
                pose_correctness_scores[0] if pose_correctness_scores else None,  # Use the first score as default
                pose_correctness_grades[0] if repetitions > 0 else "No Prediction",
            )

            # Save angle correctness feedback with time of appearance
            if repetitions > 0:
                for feedback in angle_correctness:
                    self.controller.db.insert_angle_correctness(
                        exercise_session_id,
                        feedback.get("body_part"),
                        feedback.get("angle"),
                        feedback.get("comment"),
                        time_of_appearance=time_of_appearance  # Pass time of appearance
                    )

            # Save each score and grade as separate entries in pose_correctness
            if repetitions > 0:
                for score, grade in zip(pose_correctness_scores, pose_correctness_grades):
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
                    "pose_correctness_score": pose_correctness_scores[0] if pose_correctness_scores else "N/A",
                    "pose_correctness_grade": "\n".join(
                        pose_correctness_grades) if repetitions > 0 else "No Prediction",
                }
            )

    def extract_feedback_components(self, feedback_list):
        """Extract feedback components from the feedback list."""
        angle_correctness = []
        pose_correctness_scores = []
        pose_correctness_grades = []

        for feedback in feedback_list:
            if isinstance(feedback, dict):
                if "angle_feedback" in feedback:
                    for msg in feedback["angle_feedback"]:
                        if msg.lower() == "good form!":
                            continue
                        try:
                            body_part, rest = msg.split(":")
                            angle, comment = rest.split("\n")
                            angle_correctness.append({
                                "body_part": body_part.strip(),
                                "angle": float(angle.replace("°", "").strip()),
                                "comment": comment.strip("()")
                            })
                        except ValueError:
                            pass  # Skip if parsing fails

                if "pose_feedback" in feedback:
                    for msg in feedback["pose_feedback"]:
                        if "Score:" in msg:
                            try:
                                score = float(msg.split("Score:")[1].strip())
                                pose_correctness_scores.append(score)
                            except ValueError:
                                pass
                        else:
                            pose_correctness_grades.append(msg.strip())  # Ensure full strings


        return angle_correctness, pose_correctness_scores, pose_correctness_grades

    def display_analysis(self, analysis_results, feedback_results):
        """Display the analysis results and feedback in the text area."""
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, "Exercise Analysis:\n")
        for result in analysis_results:
            display_text = (
                f"Exercise: {result['exercise']}\n"
                f"Start Time: {result['start_time']}\n"
                f"End Time: {result['end_time']}\n"
                f"Average Confidence: {result['confidence']}%\n\n"
            )
            self.result_text.insert(tk.END, display_text)

        # Display feedback
        self.result_text.insert(tk.END, "\nDetailed Feedback:\n")
        for feedback in feedback_results:
            display_text = (
                f"Timestamp: {feedback['timestamp']}\n"
                f"Exercise: {feedback['exercise']}\n"
                f"Angle Correctness:\n"
            )
            for angle_msg in feedback["angle_feedback"]:
                display_text += f" - {angle_msg}\n"
            display_text += "Pose Correctness:\n"
            for pose_msg in feedback["pose_feedback"]:
                display_text += f" - {pose_msg}\n"
            display_text += "\n"
            self.result_text.insert(tk.END, display_text)
