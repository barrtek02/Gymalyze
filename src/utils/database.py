import sqlite3
from sqlite3 import Error
import os

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/app_data.db")


class Database:
    def __init__(self, db_path=None):
        """Initialize the database connection. Default to DB_PATH if none provided."""
        self.db_path = db_path or DB_PATH  # Use provided path or default path
        self.conn = self.create_connection()

    def create_connection(self):
        """Create a database connection to the SQLite database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
        except Error as e:
            print(f"Error connecting to the database: {e}")
        return conn

    def create_table(self):
        """Create tables for users and performance tracking"""
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        );
        """
        create_performance_table = """
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            exercise_name TEXT,
            reps INTEGER,
            accuracy_score REAL,
            date TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_users_table)
            cursor.execute(create_performance_table)
            self.conn.commit()
        except Error as e:
            print(f"Error creating tables: {e}")

    def insert_user(self, username, password):
        """Insert a new user into the users table"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password),
            )
            self.conn.commit()
        except Error as e:
            print(f"Error inserting user: {e}")

    def validate_user(self, username, password):
        """Validate user credentials"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?", (username, password)
        )
        return cursor.fetchone()

    def insert_performance(self, user_id, exercise_name, reps, accuracy_score, date):
        """Insert performance data into the performance table"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
            INSERT INTO performance (user_id, exercise_name, reps, accuracy_score, date) 
            VALUES (?, ?, ?, ?, ?)""",
                (user_id, exercise_name, reps, accuracy_score, date),
            )
            self.conn.commit()
        except Error as e:
            print(f"Error inserting performance data: {e}")

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
