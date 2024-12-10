import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch


class ExerciseEvaluator:
    # Define class-level attributes for the JSON file paths
    ANGLE_THRESHOLDS_PATH = Path(
        os.path.join(os.path.dirname(__file__), "../data/angle_thresholds.json")
    )
    FEEDBACK_PATH = Path(
        os.path.join(os.path.dirname(__file__), "../data/body_parts_and_feedback.json")
    )

    FRIENDLY_ANGLE_NAMES = {
        "Left Hip-Left Knee-Left Ankle": "Knee Angle (Left)",
        "Right Hip-Right Knee-Right Ankle": "Knee Angle (Right)",
        "Left Shoulder-Left Hip-Left Knee": "Torso Angle (Left)",
        "Right Shoulder-Right Hip-Right Knee": "Torso Angle (Right)",
        "Left Shoulder-Left Elbow-Left Wrist": "Arm Alignment (Left)",
        "Right Shoulder-Right Elbow-Right Wrist": "Arm Alignment (Right)",
        "Left Shoulder-Left Hip-Left Ankle": "Full Body Alignment (Left)",
        "Right Shoulder-Right Hip-Right Ankle": "Full Body Alignment (Right)",
    }

    def __init__(self):
        """
        Initialize the ExerciseEvaluator by loading thresholds and feedback rules.
        """
        # Load angle thresholds and feedback rules
        with self.ANGLE_THRESHOLDS_PATH.open("r") as f:
            self.angle_thresholds = json.load(f)

        with self.FEEDBACK_PATH.open("r") as f:
            feedback_data = json.load(f)
            self.body_parts = feedback_data["body_parts"]
            self.feedback_rules = feedback_data["feedback_rules"]

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate the angle formed by three 3D points: a, b, c."""
        a, b, c = map(torch.tensor, (a, b, c))

        ba = a - b  # Vector from b to a
        bc = c - b  # Vector from b to c
        cos_angle = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc) + 1e-7)
        angle = torch.acos(
            torch.clamp(cos_angle, -1.0, 1.0)
        )  # Clamp for numerical stability
        return angle

    def convert_angle_name_to_friendly(self, angle_name: str) -> str:
        """
        Convert an angle name like "Left Shoulder-Left Elbow-Left Wrist"
        into a user-friendly format like "Arm Alignment".
        """
        return self.FRIENDLY_ANGLE_NAMES.get(angle_name, angle_name)

    def evaluate_exercise(
        self, exercise: str, input_data: np.ndarray, reconstructed_data: np.ndarray
    ):
        """
        Evaluate an exercise using a subset of representative frames.

        Args:
            exercise (str): The name of the exercise being evaluated (e.g., "squat").
            input_data (np.ndarray): The input pose data for the sequence (shape: [frames, landmarks, 3]).
            reconstructed_data (np.ndarray): The reconstructed pose data (shape: [frames, landmarks, 3]).

        Returns:
            pd.DataFrame: Frame-by-frame evaluation results.
            float: Overall reconstruction error as a percentage.
        """
        # Extract relevant angles and thresholds
        if input_data.shape[1] == 132:  # 33 landmarks * 4 features
            input_data = input_data.reshape(input_data.shape[0], 33, 4)
        if reconstructed_data.shape[1] == 132:  # 33 landmarks * 4 features
            reconstructed_data = reconstructed_data.reshape(
                reconstructed_data.shape[0], 33, 4
            )

        relevant_angles = self.feedback_rules[exercise]
        thresholds = self.angle_thresholds[exercise]

        total_frames = input_data.shape[0]

        # Select 5 evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, num=5, dtype=int)

        frame_results = []
        unique_feedback = set()  # To track already added feedback
        total_deviation = 0

        for frame_idx in frame_indices:
            input_frame = input_data[frame_idx]
            reconstructed_frame = reconstructed_data[frame_idx]
            frame_deviation = 0

            for angle_name, feedback_text in relevant_angles.items():
                # Convert the angle name to a user-friendly format
                friendly_angle_name = self.convert_angle_name_to_friendly(angle_name)

                # Parse angle points from the angle name
                angle_parts = angle_name.split("-")
                if len(angle_parts) != 3:
                    raise ValueError(f"Invalid angle name: {angle_name}")

                a, b, c = angle_parts
                idx_a, idx_b, idx_c = (
                    self.body_parts.index(a),
                    self.body_parts.index(b),
                    self.body_parts.index(c),
                )

                # Extract 3D coordinates of landmarks
                point_a = input_frame[idx_a]
                point_b = input_frame[idx_b]
                point_c = input_frame[idx_c]

                reconstructed_a = reconstructed_frame[idx_a]
                reconstructed_b = reconstructed_frame[idx_b]
                reconstructed_c = reconstructed_frame[idx_c]

                # Calculate angles
                input_angle = self.calculate_angle(point_a, point_b, point_c)
                reconstructed_angle = self.calculate_angle(
                    reconstructed_a, reconstructed_b, reconstructed_c
                )

                deviation = abs(input_angle - reconstructed_angle)
                frame_deviation += deviation.item()

                if deviation.item() > thresholds[angle_name]:  # Compare to threshold
                    if feedback_text not in unique_feedback:  # Avoid duplicates
                        unique_feedback.add(feedback_text)
                        frame_results.append(
                            {
                                "Frame": frame_idx,
                                "Friendly Angle": friendly_angle_name,
                                "Input Angle (degrees)": np.degrees(input_angle.item()),
                                "Reconstructed Angle (degrees)": np.degrees(
                                    reconstructed_angle.item()
                                ),
                                "Deviation (degrees)": np.degrees(deviation.item()),
                                "Threshold (degrees)": np.degrees(
                                    thresholds[angle_name]
                                ),
                                "Feedback": feedback_text,
                            }
                        )

            total_deviation += frame_deviation

        # Calculate overall reconstruction error as a percentage
        overall_reconstruction_error = (
            total_deviation / (len(frame_indices) * len(relevant_angles))
        ) * 100

        return pd.DataFrame(frame_results), overall_reconstruction_error
