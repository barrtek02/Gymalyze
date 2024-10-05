from src.utils.pose_estimator import PoseEstimator
from src.utils.video_processor import VideoProcessor
from pathlib import Path

if __name__ == "__main__":

    # Initialize your pose estimator and video processor
    video_processor = VideoProcessor()

    # Define the input directory containing videos and the output directory to save results
    input_dir = Path(
        r"C:\Users\barrt\PycharmProjects\Gymalyze\src\data\videos\verified_data\verified_data\data_btc_10s"
    )
    output_dir = Path(
        r"C:\Users\barrt\PycharmProjects\Gymalyze\src\data\videos\output_landmarks"
    )

    # Process all videos in the directory and save results after each video
    video_processor.process_videos_in_directory(input_dir, output_dir)
