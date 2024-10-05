from pathlib import Path

import numpy as np


def merge_npy_files(
    directory_path: Path, output_X_file: Path, output_y_file: Path
) -> None:
    """
    Merge all separate landmarks and label .npy files into single .npy files.

    :param directory_path: Path to the directory containing .npy files.
    :param output_X_file: Path to the output landmarks .npy file.
    :param output_y_file: Path to the output labels .npy file.
    """
    X_list = []
    y_list = []

    for file in directory_path.glob("*_landmarks.npy"):
        X_list.append(np.load(file, allow_pickle=True))

    for file in directory_path.glob("*_labels.npy"):
        y_list.append(np.load(file, allow_pickle=True))

    X_combined = np.concatenate(X_list)
    y_combined = np.concatenate(y_list)

    np.save(output_X_file, X_combined)
    np.save(output_y_file, y_combined)


if __name__ == "__main__":
    input_dir = Path(
        r"C:\Users\barrt\PycharmProjects\Gymalyze\src\data\videos\output_landmarks"
    )
    output_X_file = Path(
        r"C:\Users\barrt\PycharmProjects\Gymalyze\src\data\pose_data.npy"
    )
    output_y_file = Path(
        r"C:\Users\barrt\PycharmProjects\Gymalyze\src\data\pose_labels.npy"
    )

    merge_npy_files(input_dir, output_X_file, output_y_file)
