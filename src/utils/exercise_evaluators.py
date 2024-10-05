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


class LatPulldownEvaluator:
    def __init__(self):
        self.elbow_angle_range = (80, 120)

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

        return feedback if feedback else ["Good lat pulldown form!"]


evaluators = {
    "squat": SquatEvaluator(),
    "deadlift": DeadliftEvaluator(),
    "bench press": BenchPressEvaluator(),
    "push-up": PushUpEvaluator(),
    "lat pulldown": LatPulldownEvaluator(),
}

cap = cv2.VideoCapture(0)  # Open the webcam

sequence_length = 30  # Adjust as per your training
sequence_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find pose landmarks
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        keypoints = extract_keypoints(results.pose_landmarks)
        sequence_buffer.append(keypoints)

        if len(sequence_buffer) == sequence_length:
            # Prepare the sequence
            sequence = preprocess_sequence(sequence_buffer)
            sequence = np.expand_dims(sequence, axis=0)
            sequence = torch.tensor(sequence, dtype=torch.float32).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(sequence)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()

            # Get predicted exercise name
            predicted_exercise = class_names[predicted_class_idx]

            # Evaluate correctness
            evaluator = evaluators[predicted_exercise]
            feedback = evaluator.evaluate(results.pose_landmarks.landmark)

            # Display feedback
            feedback_text = "; ".join(feedback)
            cv2.putText(
                frame,
                f"{predicted_exercise.capitalize()} ({confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                feedback_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # Maintain sequence buffer
            sequence_buffer.pop(0)
    else:
        # Clear buffer if no pose is detected
        sequence_buffer = []

    cv2.imshow("Exercise Evaluation", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
