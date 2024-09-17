import tkinter as tk


class HomeScreen(tk.Frame):
    def __init__(self, parent, controller, db):
        super().__init__(parent)
        self.controller = controller
        self.db = db

        tk.Label(self, text="Bodybuilding Trainer Home", font=("Arial", 16)).pack(
            pady=20
        )

        # Start Live Detection Button
        live_button = tk.Button(
            self,
            text="Start Live Detection",
            command=lambda: controller.show_frame("LiveDetectionScreen"),  # Fixed here
            width=30,
            height=2,
        )
        live_button.pack(pady=10)

        train_button = tk.Button(
            self,
            text="Train Specific Exercise",
            command=lambda: controller.show_frame("TrainSpecificScreen"),  # Fixed here
            width=30,
            height=2,
        )
        train_button.pack(pady=10)

        upload_button = tk.Button(
            self,
            text="Upload Video for Review",
            command=lambda: controller.show_frame("UploadVideoScreen"),
            width=30,
            height=2,
        )
        upload_button.pack(pady=10)

        logout_button = tk.Button(
            self,
            text="Logout",
            command=lambda: controller.show_frame("LoginScreen"),
            width=30,
            height=2,
        )
        logout_button.pack(pady=10)
