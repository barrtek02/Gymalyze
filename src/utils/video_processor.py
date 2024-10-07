import cv2
import torch

from src.utils.pose_estimator import PoseEstimator
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class VideoProcessor:
    @staticmethod
    def show_video_with_skeleton(video_path: Path) -> None:
        """
        Display the video with the pose skeleton overlaid on the frames in real-time.

        :param video_path: Path to the video file (as a Path object).
        """
        cap = cv2.VideoCapture(str(video_path))
        pose_estimator = PoseEstimator()

        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose_landmarks = pose_estimator.estimate_pose(frame)

            if pose_landmarks:
                frame = pose_estimator.draw_pose(frame, pose_landmarks)

            cv2.imshow("Video with Pose Skeleton", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def process_video(video_path: Path) -> np.ndarray:
        """
        Process a single video file and extract pose landmarks for each frame.

        :param video_path: Path to the video file (as a Path object).
        :return: A NumPy array of pose landmarks for each frame.
        """
        pose_estimator = PoseEstimator()
        video_capture = VideoProcessor._open_video(video_path)
        all_landmarks = VideoProcessor._extract_landmarks(video_capture, pose_estimator)
        video_capture.release()
        return VideoProcessor.format_landmarks(all_landmarks)

    @staticmethod
    def _open_video(video_path: Path) -> cv2.VideoCapture:
        """
        Open the video file for processing.

        :param video_path: Path to the video file (as a Path object).
        :return: An opened VideoCapture object.
        """
        video_capture = cv2.VideoCapture(str(video_path))
        if not video_capture.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        return video_capture

    @staticmethod
    def _extract_landmarks(
        video_capture: cv2.VideoCapture, pose_estimator
    ) -> list[list[np.ndarray]]:
        """
        Read video frames and extract pose landmarks from each frame.

        :param video_capture: Opened VideoCapture object.
        :param pose_estimator: PoseEstimator instance.
        :return: A list of pose landmarks for each frame.
        """
        all_landmarks = []

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            try:
                pose_landmarks = VideoProcessor.process_frame(frame, pose_estimator)
                if pose_landmarks:
                    all_landmarks.append(pose_landmarks)
            except ValueError as e:
                print(f"Skipping frame due to error: {e}")
                continue

        return all_landmarks

    @staticmethod
    def process_pose_landmarks(pose_landmarks) -> list[np.ndarray]:
        """
        Process the pose landmarks into a format suitable for training the model.
        :param pose_landmarks:
        :return:
        """
        return [
            np.array([lm.x, lm.y, lm.z, lm.visibility]).flatten()
            for lm in pose_landmarks.landmark
        ]

    @staticmethod
    def format_landmarks(all_landmarks: list[list[np.ndarray]]) -> np.ndarray:
        """
        Format the list of pose landmarks into a NumPy array.

        :param all_landmarks: List of pose landmarks for each frame.
        :return: A NumPy array of the formatted pose landmarks.
        """
        return np.array(
            [np.array(sample).flatten() for sample in all_landmarks], dtype=np.float32
        )

    @staticmethod
    def save_pose_data(X: list, y: list, output_dir: Path, video_name: str) -> None:
        """
        Save the processed pose landmarks and labels to .npy files.

        :param X: List of pose landmarks.
        :param y: List of corresponding labels.
        :param output_dir: Directory where the .npy files will be saved.
        :param video_name: Name of the video file (without extension) for creating unique filenames.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / f"{video_name}_landmarks.npy", np.array(X, dtype=object))
        np.save(output_dir / f"{video_name}_labels.npy", np.array(y))

    def process_videos_in_label(self, label_dir: Path, output_dir: Path) -> None:
        """
        Process all video files within a label directory, save each video's pose landmarks and labels.

        :param label_dir: Path to the directory containing video files.
        :param output_dir: Directory where the landmarks and labels will be saved.
        """
        label = label_dir.name

        video_files = list(label_dir.glob("*.mp4"))
        for video_file in tqdm(
            video_files, desc=f"Processing {label_dir.name} videos", leave=False
        ):
            video_name = video_file.stem
            landmark_file = output_dir / f"{video_name}_landmarks.npy"
            label_file = output_dir / f"{video_name}_labels.npy"

            if landmark_file.exists() and label_file.exists():
                print(f"Skipping {video_file.name} as it is already processed.")
                continue

            landmarks = self.process_video(video_file)
            labels = [label] * len(landmarks)

            self.save_pose_data(landmarks, labels, output_dir, video_name)

    def process_videos_in_directory(
        self, directory_path: Path, output_dir: Path
    ) -> None:
        """
        Process all video files in directories where each directory represents a label for the pose.
        Save the extracted pose landmarks and labels to .npy files after processing each video.

        :param directory_path: Path to the root directory containing subdirectories representing pose labels.
        :param output_dir: Directory where the landmarks and labels will be saved.
        """
        label_dirs = [f for f in directory_path.iterdir() if f.is_dir()]

        with tqdm(total=len(label_dirs), desc="Processing directories") as pbar:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self.process_videos_in_label, label_dir, output_dir
                    ): label_dir.name
                    for label_dir in label_dirs
                }

                for future in futures:
                    future.result()
                    pbar.update(1)

    @staticmethod
    def classify_sequence(
        model: torch.nn.Module, sequence: np.ndarray, device: torch.device
    ) -> np.ndarray:
        """
        Predict the exercise class for the given sequence using the model.

        :param device: PyTorch device to use for inference.
        :param model: PyTorch model for exercise classification.
        :param sequence: Numpy array containing the pose landmarks for a single video.
        :return: Numpy array containing the predicted class probabilities.
        """
        sequence = np.expand_dims(sequence, axis=0)
        sequence = torch.tensor(sequence, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(sequence)
            # Move the tensor to the CPU before converting to numpy
            return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()

    @staticmethod
    def convert_index_to_exercise_name(index: int) -> str:
        """
        Convert the predicted class index to the corresponding exercise name.

        :param index: Index of the predicted class.
        :return: Exercise name corresponding to the index.
        """
        exercise_names = [
            "bench_press",
            "bicep_curl",
            "squat",
            "deadlift",
            "push_up",
        ]

        return exercise_names[index]
