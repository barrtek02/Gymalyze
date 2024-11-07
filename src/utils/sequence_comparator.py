import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


class SequenceComparator:
    """
    A class to compare a user-provided sequence against sample sequences from a dataset
    using the Hungarian algorithm for alignment and cosine similarity for similarity measurement.
    """

    def __init__(
        self, dataset: np.ndarray, labels: np.ndarray, label_to_exercise: Dict[int, str]
    ):
        """
        Initialize the SequenceComparator with the dataset, labels, and specified indices for each exercise.

        :param dataset: NumPy array of shape (n_samples, n_frames, n_features).
        :param labels: NumPy array of shape (n_samples,).
        :param label_to_exercise: Dictionary mapping label indices to exercise names.
        """
        self.dataset = dataset
        self.labels = labels
        self.label_to_exercise = label_to_exercise

        # Updated expert-selected indices based on new data provided
        self.expert_selected_indices = {
            "bench_press": [544, 748, 623],
            "bicep_curl": [364, 385, 699],
            "squat": [45, 422, 215],
            "deadlift": [733, 536, 692],
            "push_up": [56, 420, 129],
        }

        # Precompute sequences based on expert-selected indices
        self.sample_sequences = {
            exercise_name: self.get_sequences_by_indices(indices)
            for exercise_name, indices in self.expert_selected_indices.items()
        }

    def get_sequences_by_indices(self, indices: List[int]) -> List[np.ndarray]:
        """
        Retrieve sample sequences based on specified indices.

        :param indices: List of indices specifying the sequences to retrieve.
        :return: List of NumPy arrays representing sample sequences.
        """
        return [self.dataset[idx] for idx in indices]

    def align_sequences(
        self, user_sequence: np.ndarray, sample_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Align the user sequence with the sample sequence using the Hungarian algorithm.

        :param user_sequence: NumPy array of shape (n_frames_user, n_features).
        :param sample_sequence: NumPy array of shape (n_frames_sample, n_features).
        :return: Aligned sample sequence as a NumPy array of shape (n_frames_user, n_features).
        """
        user_sequence = np.asarray(user_sequence, dtype=float)
        sample_sequence = np.asarray(sample_sequence, dtype=float)

        if not np.all(np.isfinite(user_sequence)) or not np.all(
            np.isfinite(sample_sequence)
        ):
            raise ValueError(
                "One of the sequences contains non-finite values (NaN or Inf)."
            )

        cost_matrix = np.linalg.norm(
            user_sequence[:, np.newaxis, :] - sample_sequence[np.newaxis, :, :], axis=2
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        aligned_sample_sequence = sample_sequence[col_ind]

        n_frames_user = user_sequence.shape[0]
        if aligned_sample_sequence.shape[0] > n_frames_user:
            aligned_sample_sequence = aligned_sample_sequence[:n_frames_user]
        elif aligned_sample_sequence.shape[0] < n_frames_user:
            padding_frames = n_frames_user - aligned_sample_sequence.shape[0]
            last_frame = aligned_sample_sequence[-1].reshape(1, -1)
            padding = np.repeat(last_frame, padding_frames, axis=0)
            aligned_sample_sequence = np.vstack((aligned_sample_sequence, padding))

        return aligned_sample_sequence

    def compute_cosine_similarity(
        self, user_sequence: np.ndarray, aligned_sample_sequence: np.ndarray
    ) -> float:
        """
        Compute the cosine similarity between the user sequence and the aligned sample sequence.

        :param user_sequence: NumPy array of shape (n_frames, n_features).
        :param aligned_sample_sequence: NumPy array of shape (n_frames, n_features).
        :return: Cosine similarity score.
        """
        user_flat = user_sequence.flatten().reshape(1, -1)
        sample_flat = aligned_sample_sequence.flatten().reshape(1, -1)
        similarity = cosine_similarity(user_flat, sample_flat)[0][0]
        return similarity

    def compare(self, user_sequence: np.ndarray, exercise_name: str) -> float:
        """
        Compare the user sequence against the specified sample sequences for the specified exercise
        and return the average cosine similarity.

        :param user_sequence: NumPy array of shape (n_frames_user, n_features).
        :param exercise_name: Name of the exercise to compare against.
        :return: Average cosine similarity score.
        """
        if user_sequence.ndim != 2:
            raise ValueError(
                f"User sequence must be 2D. Received shape: {user_sequence.shape}"
            )

        # Retrieve precomputed sample sequences for the exercise
        sample_sequences = self.sample_sequences.get(exercise_name)
        if not sample_sequences:
            raise ValueError(f"No stored samples found for exercise '{exercise_name}'.")

        similarities = []
        for sample_seq in sample_sequences:
            if sample_seq.ndim != 2:
                raise ValueError(
                    f"Sample sequence must be 2D. Received shape: {sample_seq.shape}"
                )

            aligned_sample = self.align_sequences(user_sequence, sample_seq)
            similarity = self.compute_cosine_similarity(user_sequence, aligned_sample)
            similarities.append(similarity)

        average_similarity = np.mean(similarities) if similarities else 0.0
        return average_similarity
