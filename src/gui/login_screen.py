import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


class LoginScreen(tk.Frame):
    def __init__(self, parent, controller, db):
        super().__init__(parent)
        self.controller = controller
        self.db = db

        # Username label and entry
        ttk.Label(self, text="Username:").pack(pady=10)
        self.username_entry = ttk.Entry(self)  # Use ttk.Entry for theme consistency
        self.username_entry.pack()

        # Pre-populate the username entry with 'admin'
        self.username_entry.insert(0, "admin")

        # Password label and entry
        ttk.Label(self, text="Password:").pack(pady=10)
        self.password_entry = ttk.Entry(
            self, show="*"
        )  # Use ttk.Entry for theme consistency
        self.password_entry.pack()

        # Pre-populate the password entry with 'admin'
        self.password_entry.insert(0, "admin")

        # Login button
        login_button = ttk.Button(
            self, text="Login", command=self.check_login
        )  # Use ttk.Button
        login_button.pack(pady=20)

        # Sign Up button
        sign_up_button = ttk.Button(
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
