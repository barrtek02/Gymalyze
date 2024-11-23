import logging
import tkinter as tk
from tkinter import filedialog, Text
from datetime import timedelta, datetime
from tkinter import ttk
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import cv2
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

        # Header Label
        header_label = tk.Label(
            self, text="Upload and Analyze Video", font=("Arial", 16, "bold")
        )
        header_label.grid(row=0, column=0, pady=10)

        # Button to upload video
        self.upload_button = ttk.Button(
            self, text="Upload Video", command=self.upload_video
        )
        self.upload_button.grid(row=1, column=0, pady=10)

        # Progress bar
        self.progress_bar = Progressbar(
            self, orient="horizontal", length=400, mode="determinate"
        )
        self.progress_bar.grid(row=2, column=0, pady=10)

        # Video name display
        self.video_label = tk.Label(self, text="No video uploaded", font=("Arial", 12))
        self.video_label.grid(row=3, column=0, pady=10)

        # Frame preview area
        self.image_label = tk.Label(
            self, text="No preview available", font=("Arial", 12)
        )
        self.image_label.grid(row=4, column=0, pady=10)

        # Text area to display results
        self.result_text = Text(
            self, height=20, width=80, wrap="word", state="disabled"
        )
        self.result_text.grid(row=5, column=0, padx=10, pady=10)

        # Return button to go back to the home screen
        self.return_button = ttk.Button(
            self,
            text="Return to Home",
            command=lambda: controller.show_frame("HomeScreen"),
        )
        self.return_button.grid(row=6, column=0, pady=10)

    def upload_video(self):
        """Open file dialog for user to select a video."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )

        if file_path:
            # Update UI with video name
            self.video_label.config(text=f"Processing: {file_path.split('/')[-1]}")

            # Display the first frame
            first_frame = self.get_first_frame(file_path)
            if first_frame:
                self.show_image(first_frame)

            # Start progress bar and processing
            self.progress_bar["value"] = 0
            self.update_idletasks()

            # Start session time tracking
            self.session_start_time = datetime.now()

            # Simulate processing in steps
            self.after(500, lambda: self.process_video(file_path))

    def process_video(self, file_path):
        """Simulate video processing and display results."""
        self.progress_bar["value"] = 50
        self.update_idletasks()

        try:
            # Process video and get analysis results
            analysis_results, feedback_results = (
                self.frame_processor.process_uploaded_video(file_path)
            )

            # Save session and analysis data
            self.save_session_data()
            self.save_analysis_data(analysis_results, feedback_results)

        except Exception as e:
            print(f"Error during processing: {e}")

        # Finalize progress bar
        self.progress_bar["value"] = 100
        self.update_idletasks()

        # Display results
        self.display_analysis(analysis_results, feedback_results)

    def get_first_frame(self, video_path):
        """Extract the first frame from the video using OpenCV."""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                image = Image.fromarray(frame)
                image = image.resize(
                    (200, 150), Image.Resampling.LANCZOS
                )  # Resize for preview
                return image
        except Exception as e:
            print(f"Error extracting first frame: {e}")
        return None

    def show_image(self, image):
        """Display an image in the image_label."""
        img_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference to avoid garbage collection

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
        self.result_text.configure(state="normal")
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
        self.result_text.configure(state="disabled")
