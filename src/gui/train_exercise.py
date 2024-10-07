import tkinter as tk


class TrainSpecificScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Label(self, text="Train Specific Exercise", font=("Arial", 16)).pack(pady=20)

        # Placeholder for specific exercise selection and feedback
        tk.Label(self, text="Exercise-specific feedback would go here.").pack(pady=10)

        # Back to Home button
        back_button = tk.Button(
            self,
            text="Back to Home",
            command=lambda: controller.show_frame("HomeScreen"),
            width=30,
            height=2,
        )
        back_button.pack(pady=20)
