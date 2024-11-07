import threading
from datetime import timedelta

import cv2
import numpy as np

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


class FrameProcessor:
    def __init__(self, controller):
        self.controller = controller  # To access classification model and device

        self.current_similarity = None
        self.fps: FPS | None = None
        self.vs: WebcamVideoStream | None = None
        self.pose_estimator: PoseEstimator | None = None
        self.video_processor: VideoProcessor = VideoProcessor()
        self.repetition_counter = RepetitionCounter()
        self.sliding_window = []
        self.sliding_window_size = 50
        self.current_prediction = "No Prediction"
        self.current_probability = 0.0
        self.current_landmarks = None
        self.frame_count = 0
        self.current_feedback = [
            ("Angle Correctness", ["No Feedback"]),
            ("Pose Correctness", ["Score: 0.0", "No Prediction"]),
        ]
        # Thread control variables
        self.stop_event = threading.Event()
        self.process_thread = None

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

    def start(self):
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

    def stop(self):
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

    def get_current_frame(self):
        """Return the current frame with pose drawn."""
        if self.vs is None:
            return None
        frame = self.vs.read()
        if (
            frame is not None
            and self.pose_estimator is not None
            and self.current_landmarks is not None
        ):
            self.pose_estimator.draw_pose(frame, self.current_landmarks)
        if self.fps is not None:
            self.fps.update()
        return frame

    def get_current_data(self):
        """Return data needed by the GUI."""
        data = {
            "fps": int(self.fps.fps()) if self.fps else 0,
            "current_prediction": self.current_prediction,
            "current_probability": self.current_probability,
            "repetition_count": self.repetition_counter.get_repetition_count(
                self.current_prediction
            ),
            "current_feedback": self.current_feedback,
        }
        return data

    def process_uploaded_video(self, video_path: str):
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

        return analysis_results  # Return the analysis results

    def classify_sequence(self, sequence: np.ndarray) -> tuple[int, float]:
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
        analysis_results: list,
    ) -> None:
        """Add the classified exercise with timestamps and average confidence to the results list."""
        result = {
            "exercise": f"Class {self.video_processor.convert_index_to_exercise_name(exercise_class)}",
            "start_time": timedelta(seconds=int(start_time)),
            "end_time": timedelta(seconds=int(end_time)),
            "confidence": round(avg_confidence, 2),
        }
        analysis_results.append(result)
