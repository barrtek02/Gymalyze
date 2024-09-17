import tkinter as tk
from tkinter import messagebox


class LoginScreen(tk.Frame):
    def __init__(self, parent, controller, db):
        super().__init__(parent)
        self.controller = controller
        self.db = db

        tk.Label(self, text="Username:").pack(pady=10)
        self.username_entry = tk.Entry(self)
        self.username_entry.pack()

        tk.Label(self, text="Password:").pack(pady=10)
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.pack()

        login_button = tk.Button(self, text="Login", command=self.check_login)
        login_button.pack(pady=20)

        # Add a "Sign Up" button
        sign_up_button = tk.Button(
            self, text="Sign Up", command=lambda: controller.show_frame("SignUpScreen")
        )
        sign_up_button.pack(pady=10)

    def check_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        user = self.db.validate_user(username, password)
        if user:
            self.controller.show_frame("HomeScreen")
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")
