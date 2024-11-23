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
        self.configure_styles()

        # Configure grid layout
        self.configure_grid()

        # Main title label
        title_label = ttk.Label(
            self,
            text="Bodybuilding Trainer Home",
            style="Title.TLabel",
            anchor="center",
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=20, sticky="nsew")

        # Load assets
        assets_path = Path(os.path.dirname(__file__)).parent / "assets"
        self.live_photo = self.load_image(
            assets_path / "live_detection.jpg", (300, 300)
        )
        self.upload_photo = self.load_image(
            assets_path / "upload_video.jpg", (300, 300)
        )
        self.library_photo = self.load_image(
            assets_path / "database_view.jpg", (200, 200)
        )

        # Buttons with styled images and tooltips
        self.create_buttons()

    def configure_styles(self):
        """Configure ttk styles for consistent and modern UI."""
        self.style.configure(
            "TButton",
            font=("Roboto", 14, "bold"),
            padding=10,
            background="#ffffff",
            foreground="#333333",
            borderwidth=1,
        )
        self.style.map(
            "TButton",
            background=[("active", "#4CAF50"), ("pressed", "#388E3C")],
            foreground=[("active", "#ffffff")],
        )
        self.style.configure(
            "Title.TLabel",
            font=("Roboto", 24, "bold"),
            foreground="#2E7D32",
            anchor="center",
            padding=20,
        )
        self.style.configure(
            "Description.TLabel",
            font=("Roboto", 16),
            foreground="#757575",
        )

    def configure_grid(self):
        """Configure grid layout for responsiveness."""
        for col in range(2):
            self.grid_columnconfigure(col, weight=1)
        for row in range(3):
            self.grid_rowconfigure(row, weight=1)

    def load_image(self, image_path, size):
        """Load and resize an image."""
        image = Image.open(image_path).resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def create_buttons(self):
        """Create buttons with associated images and commands."""
        # Start Live Detection Button
        live_button = ttk.Button(
            self,
            text="Start Live Detection",
            image=self.live_photo,
            compound="top",
            style="TButton",
            command=lambda: self.controller.show_frame("LiveDetectionScreen"),
        )
        live_button.grid(
            row=1, column=0, padx=20, pady=20, ipadx=20, ipady=20, sticky="nsew"
        )

        # Upload Video for Review Button
        upload_button = ttk.Button(
            self,
            text="Upload Video for Review",
            image=self.upload_photo,
            compound="top",
            style="TButton",
            command=lambda: self.controller.show_frame("UploadVideoScreen"),
        )
        upload_button.grid(
            row=1, column=1, padx=20, pady=20, ipadx=20, ipady=20, sticky="nsew"
        )

        # Exercise Library Button
        library_button = ttk.Button(
            self,
            text="Exercise Library",
            image=self.library_photo,
            compound="top",
            style="TButton",
            command=lambda: self.controller.show_frame("ExerciseLibraryScreen"),
        )
        library_button.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=20,
            pady=20,
            ipadx=20,
            ipady=20,
            sticky="nsew",
        )

        # Add tooltips for better UX
        self.add_tooltip(live_button, "Start detecting exercises in real time.")
        self.add_tooltip(upload_button, "Upload a video to analyze your form.")
        self.add_tooltip(library_button, "View training history.")

    def add_tooltip(self, widget, text):
        """Add a tooltip to a widget."""
        tooltip = tk.Toplevel(widget, bg="#000000", padx=5, pady=5)
        tooltip.withdraw()  # Hide initially
        tooltip.overrideredirect(True)  # Remove window decorations
        label = ttk.Label(
            tooltip,
            text=text,
            style="Description.TLabel",
            background="#000000",
            foreground="#ffffff",
            padding=5,
        )
        label.pack()

        def show_tooltip(event):
            tooltip.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            tooltip.deiconify()

        def hide_tooltip(event):
            tooltip.withdraw()

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
