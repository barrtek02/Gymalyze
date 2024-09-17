import tkinter as tk


class LiveDetectionScreen(tk.Frame):
    def __init__(self, parent, controller, db):
        super().__init__(parent)
        self.controller = controller
        self.db = db

        tk.Label(self, text="Live Detection Mode", font=("Arial", 16)).pack(pady=20)

        tk.Label(self, text="Live camera feed would go here.").pack(pady=10)

        back_button = tk.Button(
            self,
            text="Back to Home",
            command=lambda: controller.show_frame("HomeScreen"),
            width=30,
            height=2,
        )
        back_button.pack(pady=20)