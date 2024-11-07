import tkinter as tk
from tkinter import ttk


class HomeScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Apply ttk styling
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, font=("Arial", 12))
        self.style.configure("TLabel", font=("Arial", 16, "bold"), foreground="white")

        # Configure grid layout for proper button placement
        self.grid_columnconfigure(0, weight=1)

        # Main label using ttk.Label for consistency
        ttk.Label(
            self, text="Bodybuilding Trainer Home", style="TLabel", anchor="center"
        ).grid(row=0, column=0, pady=20, sticky="n")

        # Large Start Live Detection Button
        live_button = ttk.Button(
            self,
            text="Start Live Detection",
            style="TButton",
            command=lambda: controller.show_frame("LiveDetectionScreen"),
        )
        live_button.grid(
            row=1, column=0, padx=40, pady=20, ipadx=30, ipady=20, sticky="n"
        )

        # Smaller buttons under the main button in a frame
        button_frame = ttk.Frame(self)
        button_frame.grid(row=2, column=0, pady=20, padx=40, sticky="n")

        train_button = ttk.Button(
            button_frame,
            text="Train Specific Exercise",
            style="TButton",
            command=lambda: controller.show_frame("TrainSpecificScreen"),
            width=20,
        )
        train_button.grid(row=0, column=0, padx=10, pady=10)

        upload_button = ttk.Button(
            button_frame,
            text="Upload Video for Review",
            style="TButton",
            command=lambda: controller.show_frame("UploadVideoScreen"),
            width=20,
        )
        upload_button.grid(row=0, column=1, padx=10, pady=10)
