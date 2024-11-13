# exercise_library_screen.py

import tkinter as tk
from tkinter import ttk, messagebox


class ExerciseLibraryScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Add a title label for the Exercise Library Screen
        ttk.Label(
            self,
            text="Exercise Library",
            style="TLabel",
            anchor="center",
            font=("Helvetica", 16, "bold"),
        ).pack(pady=10)

        # Frame for filter options
        filter_frame = ttk.Frame(self)
        filter_frame.pack(pady=5, fill=tk.X, padx=20)

        # Label and Entry for exercise_session_id filter
        ttk.Label(filter_frame, text="Filter by Exercise Session ID:").pack(
            side=tk.LEFT, padx=(0, 5)
        )
        self.session_id_entry = ttk.Entry(filter_frame)
        self.session_id_entry.pack(side=tk.LEFT, padx=(0, 5))

        # Button to apply filter
        filter_button = ttk.Button(
            filter_frame, text="Apply Filter", command=self.load_data
        )
        filter_button.pack(side=tk.LEFT, padx=(0, 5))

        # Button to clear filter
        clear_filter_button = ttk.Button(
            filter_frame, text="Clear Filter", command=self.clear_filter
        )
        clear_filter_button.pack(side=tk.LEFT)

        # Create a variable to keep track of the selected table
        self.selected_table = tk.StringVar(value="exercise_sessions")

        # Create a dropdown (OptionMenu) to select the table
        table_options = [
            "sessions",
            "exercise_sessions",
            "angle_correctness",
            "pose_correctness",
        ]
        table_menu = ttk.OptionMenu(
            self,
            self.selected_table,
            self.selected_table.get(),
            *table_options,
            command=self.on_table_change,
        )
        table_menu.pack(pady=5)

        # Create the Treeview widget to display data
        self.tree_frame = ttk.Frame(self)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.tree = ttk.Treeview(self.tree_frame, columns=[], show="headings")
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(
            self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind the row selection event
        self.tree.bind("<<TreeviewSelect>>", self.on_record_select)

        # Frame for displaying record details
        self.details_frame = ttk.LabelFrame(
            self, text="Record Details", padding=(10, 10)
        )
        self.details_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=(0, 10))

        self.details_text = tk.Text(
            self.details_frame, height=10, wrap=tk.WORD, state="disabled"
        )
        self.details_text.pack(fill=tk.BOTH, expand=True)

        # Buttons to view related tables
        self.view_angle_button = ttk.Button(
            self.details_frame,
            text="View Angle Correctness",
            command=self.view_angle_correctness,
        )
        self.view_angle_button.pack(pady=5)
        self.view_pose_button = ttk.Button(
            self.details_frame,
            text="View Pose Correctness",
            command=self.view_pose_correctness,
        )
        self.view_pose_button.pack(pady=5)

        # Add a back button to return to HomeScreen
        back_button = ttk.Button(
            self,
            text="Back to Home",
            style="TButton",
            command=lambda: controller.show_frame("HomeScreen"),
        )
        back_button.pack(pady=10)

        # Load data from the initially selected table
        self.load_data()

    def on_table_change(self, selected_table):
        """Callback when the selected table changes."""
        self.load_data()

    def clear_filter(self):
        """Clear the session ID filter."""
        self.session_id_entry.delete(0, tk.END)
        self.load_data()

    def on_record_select(self, event):
        """Handle row selection and display detailed information inline."""
        selected_item = self.tree.focus()
        if not selected_item:
            return  # No item selected

        # Get the item values
        item_values = self.tree.item(selected_item, "values")

        # Get the column names
        column_names = self.tree["columns"]

        # Create a dictionary of column names to values
        record = dict(zip(column_names, item_values))

        # Determine which table is currently selected
        table_name = self.selected_table.get()

        # Depending on the table, prepare relevant details
        details = ""
        if table_name == "sessions":
            details = f"Session ID: {record.get('session_id')}\nDate: {record.get('date')}\nDuration: {record.get('duration')}"
        elif table_name == "exercise_sessions":
            details = (
                f"Exercise Session ID: {record.get('exercise_session_id')}\n"
                f"Session ID: {record.get('session_id')}\n"
                f"Exercise: {record.get('exercise')}\n"
                f"Repetitions: {record.get('repetitions')}\n"
                f"Duration: {record.get('duration')}\n"
                f"Pose Correctness Score: {record.get('pose_correctness_score')}\n"
                f"Pose Correctness Grade: {record.get('pose_correctness_grade')}\n"
                f"Angle Correctness Feedback Count: {record.get('angle_correctness')}"
            )
        elif table_name == "angle_correctness":
            details = (
                f"ID: {record.get('id')}\n"
                f"Exercise Session ID: {record.get('exercise_session_id')}\n"
                f"Body Part: {record.get('body_part')}\n"
                f"Angle: {record.get('angle')}\n"
                f"Comment: {record.get('comment')}\n"
                f"Timestamp: {record.get('timestamp')}"
            )
        elif table_name == "pose_correctness":
            details = (
                f"ID: {record.get('id')}\n"
                f"Exercise Session ID: {record.get('exercise_session_id')}\n"
                f"Score: {record.get('score')}\n"
                f"Grade: {record.get('grade')}\n"
                f"Timestamp: {record.get('timestamp')}"
            )

        # Display the details in the details_text widget
        self.details_text.config(state="normal")
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, details)
        self.details_text.config(state="disabled")

    def load_data(self):
        """Load data from the selected table and display it."""
        # Get the database connection from the controller
        db = self.controller.db

        # Get the selected table name
        table_name = self.selected_table.get()

        # Get filter value
        session_id_filter = self.session_id_entry.get().strip()

        # Fetch column names
        c = db.conn.cursor()
        try:
            c.execute(f"PRAGMA table_info({table_name})")
            columns_info = c.fetchall()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrieve table info: {e}")
            return

        if not columns_info:
            messagebox.showerror("Error", f"No columns found for table '{table_name}'")
            return
        column_names = [info[1] for info in columns_info]

        # Destroy the existing Treeview columns
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
            self.tree.column(col, width=0)
        self.tree["columns"] = column_names

        # Configure the Treeview columns and headings
        for col in column_names:
            display_name = col.replace("_", " ").title()
            if table_name == "exercise_sessions" and col == "angle_correctness":
                display_name = "Angle Correctness Count"
            self.tree.heading(col, text=display_name)
            self.tree.column(col, width=150, anchor="center")

        # Construct the SQL query
        if session_id_filter:
            # Determine the appropriate column for filtering based on the table
            if table_name in [
                "exercise_sessions",
                "angle_correctness",
                "pose_correctness",
            ]:
                filter_column = "exercise_session_id"
            elif table_name == "sessions":
                filter_column = "session_id"
            else:
                filter_column = None

            if filter_column:
                query = f"SELECT * FROM {table_name} WHERE {filter_column} = ?"
                params = (session_id_filter,)
            else:
                query = f"SELECT * FROM {table_name}"
                params = ()
        else:
            query = f"SELECT * FROM {table_name}"
            params = ()

        try:
            c.execute(query, params)
            rows = c.fetchall()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrieve data: {e}")
            return

        # Clear existing data in the Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert data into the Treeview
        for row in rows:
            # Convert all None values to 'N/A' for display
            display_row = [item if item is not None else "N/A" for item in row]
            self.tree.insert("", tk.END, values=display_row)

        # Clear the details section when new data is loaded
        self.details_text.config(state="normal")
        self.details_text.delete(1.0, tk.END)
        self.details_text.config(state="disabled")

    def view_angle_correctness(self):
        """View angle_correctness table for the selected exercise_session."""
        selected_item = self.tree.focus()
        if not selected_item:
            messagebox.showwarning("Warning", "No record selected.")
            return

        # Get the item values
        item_values = self.tree.item(selected_item, "values")
        column_names = self.tree["columns"]
        record = dict(zip(column_names, item_values))

        if self.selected_table.get() != "exercise_sessions":
            messagebox.showwarning(
                "Warning",
                "Please select an exercise session to view angle correctness.",
            )
            return

        exercise_session_id = record.get("exercise_session_id")
        if not exercise_session_id:
            messagebox.showwarning("Warning", "Invalid exercise session ID.")
            return

        # Set the table to angle_correctness and apply filter
        self.selected_table.set("angle_correctness")
        self.session_id_entry.delete(0, tk.END)
        self.session_id_entry.insert(0, str(exercise_session_id))
        self.load_data()

    def view_pose_correctness(self):
        """View pose_correctness table for the selected exercise_session."""
        selected_item = self.tree.focus()
        if not selected_item:
            messagebox.showwarning("Warning", "No record selected.")
            return

        # Get the item values
        item_values = self.tree.item(selected_item, "values")
        column_names = self.tree["columns"]
        record = dict(zip(column_names, item_values))

        if self.selected_table.get() != "exercise_sessions":
            messagebox.showwarning(
                "Warning", "Please select an exercise session to view pose correctness."
            )
            return

        exercise_session_id = record.get("exercise_session_id")
        if not exercise_session_id:
            messagebox.showwarning("Warning", "Invalid exercise session ID.")
            return

        # Set the table to pose_correctness and apply filter
        self.selected_table.set("pose_correctness")
        self.session_id_entry.delete(0, tk.END)
        self.session_id_entry.insert(0, str(exercise_session_id))
        self.load_data()
