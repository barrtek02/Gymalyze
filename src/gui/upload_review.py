import tkinter as tk
from tkinter import filedialog, Text, Button
from src.utils.frame_processor import FrameProcessor


class UploadVideoScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Initialize the FrameProcessor
        self.frame_processor = FrameProcessor(controller)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Text area to display results
        self.result_text = Text(self, height=30, width=100)
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
            # Use FrameProcessor to process the video
            analysis_results, feedback_results = self.frame_processor.process_uploaded_video(file_path)
            # Display results in the text area
            self.display_analysis(analysis_results, feedback_results)

    def display_analysis(self, analysis_results: list[dict], feedback_results: list[dict]) -> None:
        """Display the analysis results and feedback in the text area."""
        self.result_text.delete(1.0, tk.END)  # Clear previous results

        # Display exercise analysis
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
            for angle_msg in feedback['angle_feedback']:
                display_text += f" - {angle_msg}\n"
            display_text += "Pose Correctness:\n"
            for pose_msg in feedback['pose_feedback']:
                display_text += f" - {pose_msg}\n"
            display_text += "\n"
            self.result_text.insert(tk.END, display_text)
