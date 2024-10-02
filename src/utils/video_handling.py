import cv2
from src.utils.pose_estimator import PoseEstimator
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, pose_estimator: PoseEstimator) -> None:
        """
        Initialize the video processor with a PoseEstimator instance.
        """
        self.pose_estimator = pose_estimator

    def show_video_with_skeleton(self, video_path: Path) -> None:
        """
        Display the video with the pose skeleton overlaid on the frames in real-time.

        :param video_path: Path to the video file (as a Path object).
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose_landmarks = self.pose_estimator.estimate_pose(frame)

            if pose_landmarks:
                frame = self.pose_estimator.draw_pose(frame, pose_landmarks)

            cv2.imshow("Video with Pose Skeleton", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_video(self, video_path: Path) -> list:
        """
        Process a single video file, extract pose landmarks, and return them.

        :param video_path: Path to the video file (as a Path object).
        :return: A list of pose landmarks for each frame in the video.
        """
        pose_estimator = (
            PoseEstimator()
        )  # Create a new PoseEstimator instance for each thread
        cap = cv2.VideoCapture(str(video_path))
        all_landmarks = []

        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                pose_landmarks = pose_estimator.estimate_pose(frame)

                if pose_landmarks:
                    all_landmarks.append(pose_landmarks)

            except ValueError as e:
                print(f"Skipping frame due to error: {e}")
                continue

        cap.release()
        return all_landmarks

    def save_pose_data(
        self, X: list, y: list, output_dir: Path, video_name: str
    ) -> None:
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
