import numpy as np
import mediapipe as mp


class SquatEvaluator:
    def __init__(self):
        # Define the ideal ranges for joint angles
        self.knee_angle_range = (80, 100)  # Ideal knee angle at the bottom position
        self.hip_angle_range = (70, 100)  # Ideal hip angle at the bottom position
        self.back_angle_range = (160, 180)  # Ideal back angle during the movement

    def calculate_angle(self, a, b, c):
        """
        Calculate the angle at point b given three points a, b, and c.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def evaluate(self, landmarks):
        feedback = []
        mp_pose = mp.solutions.pose

        # Get coordinates
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]

        # Calculate angles
        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        back_angle = self.calculate_angle(
            left_shoulder, left_hip, [left_hip[0], left_hip[1] - 0.1]
        )

        # Evaluate knee angle
        if not (self.knee_angle_range[0] <= knee_angle <= self.knee_angle_range[1]):
            feedback.append(f"Knee angle ({knee_angle:.1f}°) is out of ideal range.")

        # Evaluate hip angle
        if not (self.hip_angle_range[0] <= hip_angle <= self.hip_angle_range[1]):
            feedback.append(f"Hip angle ({hip_angle:.1f}°) is out of ideal range.")

        # Evaluate back angle
        if not (self.back_angle_range[0] <= back_angle <= self.back_angle_range[1]):
            feedback.append(
                f"Back angle ({back_angle:.1f}°) indicates improper posture."
            )

        return feedback if feedback else ["Good squat form!"]


class DeadliftEvaluator:
    def __init__(self):
        self.back_angle_range = (160, 180)
        self.hip_angle_range = (30, 70)
        self.knee_angle_range = (160, 180)

    def calculate_angle(self, a, b, c):
        # Same as in SquatEvaluator
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def evaluate(self, landmarks):
        feedback = []
        mp_pose = mp.solutions.pose

        # Get coordinates
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]

        # Calculate angles
        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        back_angle = self.calculate_angle(
            left_shoulder, left_hip, [left_hip[0], left_hip[1] - 0.1]
        )

        # Evaluate back angle
        if not (self.back_angle_range[0] <= back_angle <= self.back_angle_range[1]):
            feedback.append(
                f"Back angle ({back_angle:.1f}°) indicates rounding. Keep your back straight."
            )

        # Evaluate hip angle
        if not (self.hip_angle_range[0] <= hip_angle <= self.hip_angle_range[1]):
            feedback.append(f"Hip angle ({hip_angle:.1f}°) is out of ideal range.")

        # Evaluate knee angle
        if not (self.knee_angle_range[0] <= knee_angle <= self.knee_angle_range[1]):
            feedback.append(
                f"Knee angle ({knee_angle:.1f}°) should be closer to 180°. Don't bend your knees too much."
            )

        return feedback if feedback else ["Good deadlift form!"]


class BenchPressEvaluator:
    def __init__(self):
        self.elbow_angle_range = (70, 110)
        self.wrist_elbow_alignment_tolerance = 0.05

    def calculate_angle(self, a, b, c):
        # Same as in SquatEvaluator
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def evaluate(self, landmarks):
        feedback = []
        mp_pose = mp.solutions.pose

        # Get coordinates
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        left_elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
        ]
        left_wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
        ]

        # Calculate elbow angle
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Evaluate elbow angle
        if not (self.elbow_angle_range[0] <= elbow_angle <= self.elbow_angle_range[1]):
            feedback.append(
                f"Elbow angle ({elbow_angle:.1f}°) indicates improper form."
            )

        # Evaluate wrist alignment
        wrist_elbow_diff = abs(left_wrist[0] - left_elbow[0])
        if wrist_elbow_diff > self.wrist_elbow_alignment_tolerance:
            feedback.append(
                "Wrist is not aligned with elbow. Keep your wrists straight."
            )

        return feedback if feedback else ["Good bench press form!"]


class PushUpEvaluator:
    def __init__(self):
        self.body_alignment_tolerance = 0.05

    def evaluate(self, landmarks):
        feedback = []
        mp_pose = mp.solutions.pose

        # Get coordinates
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]

        # Check body alignment (should be in a straight line)
        shoulder_hip_diff = abs(left_shoulder[1] - left_hip[1])
        hip_ankle_diff = abs(left_hip[1] - left_ankle[1])

        if abs(shoulder_hip_diff - hip_ankle_diff) > self.body_alignment_tolerance:
            feedback.append(
                "Body is not in a straight line. Keep your hips aligned with your shoulders and ankles."
            )

        return feedback if feedback else ["Good push-up form!"]


class BicepCurlEvaluator:
    def __init__(self):
        # Define the ideal range for elbow angles during a bicep curl
        self.elbow_angle_min = 70  # Minimum angle at the bottom of the curl
        self.elbow_angle_max = 160  # Maximum angle at the top of the curl

    def calculate_angle(self, a, b, c):
        """
        Calculate the angle at point b given three points a, b, and c.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def evaluate(self, landmarks):
        feedback = []
        mp_pose = mp.solutions.pose

        # Get coordinates for left arm
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        left_elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
        ]
        left_wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
        ]

        # Get coordinates for right arm
        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
        right_elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
        ]
        right_wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
        ]

        # Calculate angles for both arms
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(
            right_shoulder, right_elbow, right_wrist
        )

        # Evaluate left elbow angle
        if not (self.elbow_angle_min <= left_elbow_angle <= self.elbow_angle_max):
            feedback.append(
                f"Left elbow angle ({left_elbow_angle:.1f}°) is out of ideal range."
            )

        # Evaluate right elbow angle
        if not (self.elbow_angle_min <= right_elbow_angle <= self.elbow_angle_max):
            feedback.append(
                f"Right elbow angle ({right_elbow_angle:.1f}°) is out of ideal range."
            )

        return feedback if feedback else ["Good bicep curl form!"]
