import numpy as np
import mediapipe as mp


class RepetitionCounter:
    def __init__(self):
        self.exercise_counts = {
            "deadlift": 0,
            "squat": 0,
            "push_up": 0,
            "bicep_curl": 0,
            "bench_press": 0,
        }
        self.current_phase = None
        self.moving_up = False

    @staticmethod
    def get_landmark_index(exercise: str) -> int:
        """Map exercises to specific landmark indices based on movement characteristics."""
        if exercise == "deadlift":
            return mp.solutions.pose.PoseLandmark.LEFT_HIP.value
        elif exercise == "squat":
            return mp.solutions.pose.PoseLandmark.LEFT_KNEE.value
        elif exercise == "push_up":
            return mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        elif exercise == "bicep_curl":
            return mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        elif exercise == "bench_press":
            return mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        return None

    def track_phases(self, window_data, exercise: str) -> None:
        """Detect phases and count repetitions based on movement patterns for the current exercise."""
        landmark_index = self.get_landmark_index(exercise)
        if landmark_index is None:
            return

        # Extract y-coordinates of the specified landmark for phase detection
        y_positions = [landmark[landmark_index][1] for landmark in window_data]

        # Calculate dynamic thresholds for up and down phases
        avg_high = np.percentile(y_positions, 10)
        avg_low = np.percentile(y_positions, 90)
        up_threshold = avg_high + 0.05 * (avg_low - avg_high)
        down_threshold = avg_low - 0.05 * (avg_low - avg_high)

        current_position = y_positions[-1]
        if current_position <= up_threshold:
            if self.current_phase != "up":
                self.current_phase = "up"
                self.moving_up = True
        elif current_position >= down_threshold and self.moving_up:
            self.current_phase = "down"
            self.exercise_counts[exercise] += 1
            self.moving_up = False

    def get_repetition_count(self, exercise: str) -> int:
        """Return the current repetition count for the specified exercise."""
        return self.exercise_counts.get(exercise, 0)
