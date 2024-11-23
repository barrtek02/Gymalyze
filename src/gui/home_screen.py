import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from PIL import Image, ImageTk


class HomeScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Apply ttk styling
        self.style = ttk.Style()
        self.style.configure("TButton", padding=10, font=("Arial", 16))
        self.style.configure("TLabel", font=("Arial", 24, "bold"), foreground="black")

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)  # Title row
        self.grid_rowconfigure(1, weight=1)  # Buttons row
        self.grid_rowconfigure(2, weight=1)  # Third button row

        # Main label using ttk.Label
        title_label = ttk.Label(
            self, text="Bodybuilding Trainer Home", style="TLabel", anchor="center"
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=20, sticky="n")

        assets_path = Path(os.path.dirname(__file__)).parent / "assets"

        live_image = Image.open(assets_path / "live_detection.jpg").resize((300, 300))
        live_photo = ImageTk.PhotoImage(live_image)

        upload_image = Image.open(assets_path / "upload_video.jpg").resize((300, 300))
        upload_photo = ImageTk.PhotoImage(upload_image)

        library_image = Image.open(assets_path / "database_view.jpg").resize((200, 200))
        library_photo = ImageTk.PhotoImage(library_image)

        self.live_photo = live_photo
        self.upload_photo = upload_photo
        self.library_photo = library_photo

        # Start Live Detection Button
        live_button = ttk.Button(
            self,
            text="Start Live Detection",
            image=self.live_photo,
            compound="top",
            style="TButton",
            command=lambda: controller.show_frame("LiveDetectionScreen"),
        )
        live_button.grid(
            row=1,
            column=0,
            padx=20,
            pady=20,
            ipadx=20,
            ipady=20,
            sticky="n",  # Increased padding
        )

        # Upload Video for Review
        upload_button = ttk.Button(
            self,
            text="Upload Video for Review",
            image=self.upload_photo,
            compound="top",
            style="TButton",
            command=lambda: controller.show_frame("UploadVideoScreen"),
        )
        upload_button.grid(
            row=1,
            column=1,
            padx=20,
            pady=20,
            ipadx=20,
            ipady=20,
            sticky="n",
        )

        # Exercise Library Button
        library_button = ttk.Button(
            self,
            text="Exercise Library",
            image=self.library_photo,
            compound="top",
            style="TButton",
            command=lambda: controller.show_frame("ExerciseLibraryScreen"),
        )
        library_button.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=20,
            pady=20,
            ipadx=20,
            ipady=20,
            sticky="n",
        )
