import tkinter as tk
from tkinter import messagebox


class SignUpScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Label(self, text="Sign Up", font=("Arial", 16)).pack(pady=20)

        # Username entry
        tk.Label(self, text="Username:").pack(pady=10)
        self.username_entry = tk.Entry(self)
        self.username_entry.pack()

        # Password entry
        tk.Label(self, text="Password:").pack(pady=10)
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.pack()

        # Confirm password entry
        tk.Label(self, text="Confirm Password:").pack(pady=10)
        self.confirm_password_entry = tk.Entry(self, show="*")
        self.confirm_password_entry.pack()

        # Sign Up button
        sign_up_button = tk.Button(self, text="Sign Up", command=self.register_user)
        sign_up_button.pack(pady=20)

    def register_user(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        confirm_password = self.confirm_password_entry.get()

        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match")
            return

        try:
            self.controller.db.insert_user(username, password)
            messagebox.showinfo("Success", "Account created successfully!")
            self.controller.show_frame("LoginScreen")
        except Exception as e:
            messagebox.showerror("Error", f"Could not create account: {e}")
