import cv2
import mediapipe as mp
from pathlib import Path


class PoseEstimator:
    def __init__(self) -> None:
        """
        Initialize the Pose Estimator using MediaPipe.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, min_detection_confidence=0.5, model_complexity=2
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def estimate_pose(self, frame: cv2.Mat) -> mp.solutions.pose.PoseLandmark:
        """
        Estimate the pose landmarks for a given frame.

        :param frame: A single frame from a video (BGR format).
        :return: Detected pose landmarks if available, otherwise None.
        """

        # Convert the BGR frame to RGB before processing with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            return results.pose_landmarks
        return None

    def draw_pose(
        self, frame: cv2.Mat, pose_landmarks: mp.solutions.pose.PoseLandmark
    ) -> cv2.Mat:
        """
        Draw pose landmarks on the given frame.

        :param frame: The original frame (BGR format).
        :param pose_landmarks: The detected pose landmarks to be drawn on the frame.
        :return: Frame with pose landmarks drawn.
        """
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        return frame
