import logging
import tkinter as tk
from tkinter import filedialog, Text, Button
from datetime import timedelta, datetime

from src.utils.frame_processor import FrameProcessor


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
            self,
            text="Return",
            command=lambda: controller.show_frame("HomeScreen"),
            width=30,
            height=2,
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
            self.session_duration = round(
                sum(
                    (result["end_time"] - result["start_time"]).total_seconds()
                    for result in analysis_results
                ),
                2,
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

    def save_analysis_data(self, analysis_results, feedback_results):
        """Save analysis data to the database."""
        for analysis in analysis_results:
            exercise = analysis["exercise"].split()[-1]
            start_time = analysis["start_time"]
            end_time = analysis["end_time"]
            repetitions = analysis.get("repetitions", 0)  # Safely get repetitions count
            duration = (end_time - start_time).total_seconds()

            logging.debug(f"Processing analysis for exercise: {exercise}")
            logging.debug(
                f"Start Time: {start_time}, End Time: {end_time}, Duration: {duration}, Repetitions: {repetitions}"
            )

            # Extract relevant feedback for the current exercise
            print(exercise)
            print(feedback_results)
            feedback_for_exercise = [
                fb for fb in feedback_results if fb["exercise"] == exercise
            ]

            if not feedback_for_exercise:
                logging.debug(f"No feedback found for exercise: {exercise}")

            # Parse feedback for angle and pose correctness
            angle_correctness, pose_correctness_scores = (
                self.extract_feedback_components(feedback_for_exercise)
            )

            # Debug logs for traceability
            logging.debug(f"Angle Correctness Count: {len(angle_correctness)}")
            logging.debug(f"Pose Correctness Scores: {pose_correctness_scores}")

            # Save exercise session to the database
            session_id = self.controller.current_session_id
            average_pose_score = (
                round(sum(pose_correctness_scores) / len(pose_correctness_scores), 2)
                if pose_correctness_scores
                else None
            )

            exercise_session_id = self.controller.db.insert_exercise_session(
                session_id,
                exercise,
                repetitions,  # Repetition count
                round(duration, 2),
                len(angle_correctness),  # Total angles evaluated
                average_pose_score,
            )

            if exercise_session_id is None:
                logging.error(f"Failed to save session data for exercise: {exercise}")
                continue

            logging.info(
                f"Inserted exercise session with ID {exercise_session_id} for exercise: {exercise}"
            )

            # Save angle correctness feedback
            for feedback in angle_correctness:
                logging.debug(
                    f"Saving angle correctness feedback for {feedback['angle_name']}..."
                )
                self.controller.db.insert_angle_correctness(
                    exercise_session_id,
                    feedback["angle_name"],
                    round(feedback["angle"], 2),
                    round(feedback["expected_angle"], 2),
                    round(feedback["threshold"], 2),
                    feedback["comment"],
                    self.calculate_time_of_appearance(start_time),
                )

            # Save pose correctness scores
            for score in pose_correctness_scores:
                logging.debug(f"Saving pose correctness score: {score}")
                self.controller.db.insert_pose_correctness(
                    exercise_session_id,
                    round(score, 2),
                    self.calculate_time_of_appearance(start_time),
                )

    def extract_feedback_components(self, feedback_list):
        """Extract feedback components from the feedback list."""
        angle_correctness = []
        pose_correctness_scores = []

        for feedback in feedback_list:
            if isinstance(feedback, dict):
                if "frame_feedback" in feedback:
                    for frame in feedback["frame_feedback"]:
                        angle_correctness.append(
                            {
                                "angle_name": frame.get("Angle", "Unknown Angle"),
                                "angle": frame.get("Input Angle", 0.0),
                                "expected_angle": frame.get("Reconstructed Angle", 0.0),
                                "threshold": frame.get("Threshold", 0.0),
                                "comment": frame.get("Feedback", "No Feedback"),
                            }
                        )
                if "score" in feedback:
                    pose_correctness_scores.append(feedback["score"])

        return angle_correctness, pose_correctness_scores

    def calculate_time_of_appearance(self, start_time):
        """
        Calculate the time of appearance in seconds since the session started.

        Args:
            start_time (datetime): The start time of the event (as a `datetime` or `timedelta` object).

        Returns:
            float: The time difference in seconds since the session started.
        """
        if isinstance(start_time, timedelta):
            # If start_time is a timedelta, calculate relative to session start
            return round(start_time.total_seconds(), 2)
        elif isinstance(start_time, datetime):
            # If start_time is a datetime, calculate difference from session start
            return round((start_time - self.session_start_time).total_seconds(), 2)
        else:
            logging.error(f"Unexpected start_time type: {type(start_time)}")
            return 0.0

    def display_analysis(self, analysis_results, feedback_results):
        """Display the analysis results and feedback in the text area."""
        self.result_text.delete(1.0, tk.END)

        # Display exercise analysis
        self.result_text.insert(tk.END, "Exercise Analysis:\n")
        for result in analysis_results:
            display_text = (
                f"Exercise: {result['exercise']}\n"
                f"Start Time: {result['start_time']}\n"
                f"End Time: {result['end_time']}\n"
                f"Average Confidence: {result['confidence']}%\n"
                f"Repetitions: {result['repetitions']}\n\n"
            )
            self.result_text.insert(tk.END, display_text)

        # Display detailed feedback
        self.result_text.insert(tk.END, "\nDetailed Feedback:\n")
        for feedback in feedback_results:
            self.result_text.insert(
                tk.END,
                f"Timestamp: {feedback['timestamp']}\n"
                f"Exercise: {feedback['exercise']}\n",
            )

            # Add frame-by-frame feedback if there's an issue
            frame_feedback = feedback.get("frame_feedback", [])
            if frame_feedback:
                for frame in frame_feedback:
                    self.result_text.insert(
                        tk.END,
                        (
                            f"  Angle: {frame['Angle']}\n"
                            f"  Input Angle: {frame['Input Angle']:.1f}°\n"
                            f"  Reconstructed Angle: {frame['Reconstructed Angle']:.1f}°\n"
                            f"  Threshold: ±{frame['Threshold']:.1f}°\n"
                            f"  Feedback: {frame['Feedback']}\n\n"
                        ),
                    )
            else:
                self.result_text.insert(
                    tk.END, "  All angles are correct for this frame.\n\n"
                )
