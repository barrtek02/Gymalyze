import logging
from datetime import timedelta
import threading
import time
import cv2
import numpy as np
import torch
from imutils.video import FileVideoStream
from src.utils.pose_estimator import PoseEstimator
from src.utils.repetition_counter import RepetitionCounter
from src.utils.video_processor import VideoProcessor
from src.utils.imutils.video import WebcamVideoStream, FPS
from src.utils.autoencoder_exercise_evaluator import ExerciseEvaluator


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
        self.current_feedback = {
            "score": None,
            "details": [{"Feedback": "Exercise not recognized."}],
        }
        # Thread control variables
        self.stop_event = threading.Event()
        self.process_thread = None

        self.exercise_evaluator = ExerciseEvaluator()

    def start(self):
        self.stop_event.clear()  # Reset the stop event before starting the new thread

        self.current_prediction = "No Prediction"
        self.current_probability = 0.0
        self.sliding_window = []
        self.current_feedback = {
            "score": None,
            "details": [{"Feedback": "Exercise not recognized."}],
        }
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

                            # Evaluate correctness using ExerciseEvaluator
                            if (
                                self.current_prediction.lower()
                                in self.exercise_evaluator.angle_thresholds
                            ):
                                input_data = np.array(self.sliding_window)
                                reconstructed_data = self.generate_reconstructed_data(
                                    input_data
                                )
                                results_df, overall_error = (
                                    self.exercise_evaluator.evaluate_exercise(
                                        self.current_prediction.lower(),
                                        input_data,
                                        reconstructed_data,
                                    )
                                )
                                exercise_score = 100 - overall_error
                                self.current_feedback = {
                                    "score": exercise_score,
                                    "details": results_df.to_dict("records"),
                                }
                                self.current_similarity = overall_error
                            else:
                                self.current_feedback = {
                                    "score": None,
                                    "details": [
                                        {"Feedback": "Exercise not recognized."}
                                    ],
                                }
                        self.sliding_window.pop(0)
                self.frame_count += 1

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

    def generate_reconstructed_data(self, input_data: np.ndarray) -> np.ndarray:
        """
        Generates reconstructed data using the autoencoder.

        Parameters:
            input_data (np.ndarray): Input sequence of shape (frames, landmarks, features).
                                     Expected shape: (50, 33, 4).

        Returns:
            np.ndarray: Reconstructed data of the same shape as input_data.
        """
        if self.controller.autoencoder is None:
            raise ValueError("Autoencoder is not loaded!")

        # Convert input_data to a torch tensor and add batch dimension
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(
            self.controller.device
        )  # Shape: (50, 33, 4)

        # Flatten the landmarks and features: [frames, landmarks * features]
        flattened_input = input_tensor.view(
            input_tensor.size(0), -1
        )  # Shape: (50, 33 * 4)

        # Add batch dimension: [1, frames, landmarks * features]
        batched_input = flattened_input.unsqueeze(0)  # Shape: (1, 50, 33 * 4)

        # Pass through the autoencoder
        reconstructed_flat = (
            self.controller.autoencoder(batched_input).squeeze(0).detach()
        )  # Shape: (50, 33 * 4)

        # Reshape back to the original dimensions: [frames, landmarks, features]
        reconstructed_data = (
            reconstructed_flat.view(flattened_input.size(0), 33, 4).cpu().numpy()
        )  # Shape: (50, 33, 4)

        return reconstructed_data

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

    def get_current_data(self) -> dict[str, any]:
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
        """Process the uploaded video for exercise classification, evaluation, and repetition counting."""
        fvs = FileVideoStream(video_path).start()
        time.sleep(1.0)  # Allow buffer to fill

        fps = fvs.stream.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        sliding_window = []
        sliding_window_size = 50
        analysis_results = []
        feedback_results = []
        recorded_feedback_by_timestamp = {}  # Tracks feedback messages by timestamp

        current_exercise = None
        start_time = 0
        confidences = []

        pose_estimator = PoseEstimator()
        repetition_counter = RepetitionCounter()  # Initialize a new repetition counter

        while fvs.running():
            if not fvs.more():
                time.sleep(0.1)
                continue

            frame = fvs.read()
            if frame is None:
                break

            if frame_count % 4 == 0:  # Process every 4th frame
                pose_landmarks_raw = pose_estimator.estimate_pose(frame)
                if pose_landmarks_raw:
                    pose_landmarks = self.video_processor.process_pose_landmarks(
                        pose_landmarks_raw
                    )
                    sliding_window.append(pose_landmarks)

                    if len(sliding_window) == sliding_window_size:
                        sequence = np.array(
                            self.video_processor.format_landmarks(sliding_window)
                        )

                        # Classify current exercise and update confidence
                        exercise_class, confidence = self.classify_sequence(sequence)
                        confidences.append(confidence)
                        current_prediction = (
                            self.video_processor.convert_index_to_exercise_name(
                                exercise_class
                            )
                        )

                        # Detect repetitions for the current exercise
                        if current_prediction:
                            repetition_counter.track_phases(
                                sliding_window, current_prediction
                            )

                        current_time = int(
                            frame_count / fps
                        )  # Current timestamp (seconds)

                        # Save the previous exercise when it changes
                        if (
                            current_exercise is not None
                            and current_exercise != current_prediction
                        ):
                            end_time = current_time
                            avg_confidence = np.mean(confidences)
                            repetitions = repetition_counter.get_repetition_count(
                                current_exercise
                            )
                            self.add_analysis_result(
                                current_exercise,
                                start_time,
                                end_time,
                                avg_confidence,
                                analysis_results,
                                repetitions,
                            )
                            confidences = []
                            start_time = current_time

                            repetition_counter = (
                                RepetitionCounter()
                            )  # Reset the repetition counter

                        current_exercise = (
                            current_prediction  # Use the string name directly
                        )
                        sliding_window.pop(0)

                        # Evaluate the current exercise using the evaluator
                        reconstructed_data = self.generate_reconstructed_data(sequence)

                        try:
                            frame_feedback, reconstruction_error = (
                                self.exercise_evaluator.evaluate_exercise(
                                    current_prediction, sequence, reconstructed_data
                                )
                            )
                            overall_error = 100 - reconstruction_error

                            filtered_feedback = []
                            for feedback in frame_feedback.to_dict(orient="records"):
                                if (
                                    feedback["Deviation (degrees)"]
                                    > feedback["Threshold (degrees)"]
                                ):
                                    feedback_message = feedback["Feedback"]

                                    # Ensure feedback is unique for the current timestamp
                                    if (
                                        current_time
                                        not in recorded_feedback_by_timestamp
                                    ):
                                        recorded_feedback_by_timestamp[current_time] = (
                                            set()
                                        )

                                    if (
                                        feedback_message
                                        not in recorded_feedback_by_timestamp[
                                            current_time
                                        ]
                                    ):
                                        recorded_feedback_by_timestamp[
                                            current_time
                                        ].add(feedback_message)
                                        filtered_feedback.append(
                                            {
                                                "Angle": feedback["Friendly Angle"],
                                                "Input Angle": round(
                                                    feedback["Input Angle (degrees)"], 1
                                                ),
                                                "Reconstructed Angle": round(
                                                    feedback[
                                                        "Reconstructed Angle (degrees)"
                                                    ],
                                                    1,
                                                ),
                                                "Threshold": round(
                                                    feedback["Threshold (degrees)"], 1
                                                ),
                                                "Feedback": feedback_message,
                                            }
                                        )

                            if filtered_feedback:
                                feedback_results.append(
                                    {
                                        "timestamp": timedelta(seconds=current_time),
                                        "exercise": current_prediction,
                                        "frame_feedback": filtered_feedback,
                                        "score": overall_error,
                                    }
                                )
                        except KeyError:
                            logging.warning(
                                f"No evaluation rules for {current_prediction}."
                            )
                        except ValueError as e:
                            logging.error(f"Error during evaluation: {e}")

            frame_count += 1

        # Handle the last exercise
        if current_exercise is not None:
            end_time = frame_count / fps
            avg_confidence = np.mean(confidences)
            repetitions = repetition_counter.get_repetition_count(current_exercise)
            self.add_analysis_result(
                current_exercise,
                start_time,
                end_time,
                avg_confidence,
                analysis_results,
                repetitions,
            )

        fvs.stop()

        return analysis_results, feedback_results

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
        exercise: str,
        start_time: float,
        end_time: float,
        avg_confidence: float,
        analysis_results: list,
        repetitions: int,
    ) -> None:
        """Add the classified exercise with timestamps, average confidence, and repetitions to the results list."""
        result = {
            "exercise": f"Class {exercise}",
            "start_time": timedelta(seconds=int(start_time)),
            "end_time": timedelta(seconds=int(end_time)),
            "confidence": round(avg_confidence, 2),
            "repetitions": repetitions,
        }
        analysis_results.append(result)
