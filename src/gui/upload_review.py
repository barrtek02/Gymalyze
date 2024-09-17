import tkinter as tk


class UploadVideoScreen(tk.Frame):
    def __init__(self, parent, controller, db):
        super().__init__(parent)
        self.controller = controller
        self.db = db

        tk.Label(self, text="Upload Video for Review", font=("Arial", 16)).pack(pady=20)

        # Placeholder for video upload and analysis functionality
        tk.Label(self, text="Video upload and analysis would go here.").pack(pady=10)

        back_button = tk.Button(
            self,
            text="Back to Home",
            command=lambda: controller.show_frame("HomeScreen"),
            width=30,
            height=2,
        )
        back_button.pack(pady=20)
