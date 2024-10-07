import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

from src.utils.video_processor import VideoProcessor


class TestVideoProcessor(unittest.TestCase):

    @patch("cv2.VideoCapture")
    def test_open_video(self, mock_VideoCapture):
        # GIVEN
        mock_cap = MagicMock()
        mock_VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        video_processor = VideoProcessor()

        # WHEN
        video_capture = video_processor._open_video(Path("fake_video.mp4"))

        # THEN
        mock_VideoCapture.assert_called_once_with("fake_video.mp4")
        self.assertTrue(video_capture.isOpened())

    def test_process_frame(self):
        # GIVEN
        with patch("src.utils.pose_estimator.PoseEstimator") as MockPoseEstimator:
            mock_pose = MockPoseEstimator.return_value
            mock_pose.estimate_pose.return_value = MagicMock(
                landmark=[MagicMock(x=1, y=2, z=3, visibility=0.9)]
            )
            video_processor = VideoProcessor()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # WHEN
        landmarks = video_processor.process_pose_landmarks(
            mock_pose.estimate_pose(frame)
        )

        # THEN
        self.assertEqual(len(landmarks), 1)
        self.assertEqual(landmarks[0].shape, (4,))

    def test_format_landmarks(self):
        # GIVEN
        video_processor = VideoProcessor()
        all_landmarks = [
            [np.array([1.0, 2.0, 3.0, 0.9]), np.array([4.0, 5.0, 6.0, 0.8])],
            [np.array([7.0, 8.0, 9.0, 0.7]), np.array([10.0, 11.0, 12.0, 0.6])],
        ]

        # WHEN
        formatted_landmarks = video_processor.format_landmarks(all_landmarks)

        # THEN
        self.assertIsInstance(formatted_landmarks, np.ndarray)
        self.assertEqual(formatted_landmarks.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
