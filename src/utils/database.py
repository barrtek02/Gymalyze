import sqlite3
from sqlite3 import Error
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/app_data.db")


class Database:
    def __init__(self, db_path=None):
        """Initialize the database connection and create tables."""
        self.db_path = db_path or DB_PATH
        self.conn = self.create_connection()
        if self.conn:
            self.create_tables()
            self.add_time_of_appearance_column()
            self.rename_angle_correctness_table()
        else:
            logging.critical("Failed to establish database connection.")

    def create_connection(self):
        """Create a database connection to the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            logging.info(f"Connected to SQLite database at {self.db_path}")
            return conn
        except Error as e:
            logging.critical(f"Error connecting to the database: {e}")
            return None

    def create_tables(self):
        """Create tables if they don't exist."""
        try:
            c = self.conn.cursor()
            # Create sessions table
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    duration REAL
                )
            """
            )

            # Create exercise_sessions table
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS exercise_sessions (
                    exercise_session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    exercise TEXT,
                    repetitions INTEGER,
                    duration REAL,
                    angle_correctness INTEGER DEFAULT 0,
                    pose_correctness_score REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """
            )

            # Create angle_correctness table
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS angle_correctness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exercise_session_id INTEGER,
                    angle_name TEXT,
                    angle REAL,
                    expected_angle REAL,
                    threshold REAL,
                    comment TEXT,
                    time_of_appearance REAL,
                    FOREIGN KEY (exercise_session_id) REFERENCES exercise_sessions(exercise_session_id)
                )
            """
            )

            # Create pose_correctness table with rounded score
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS pose_correctness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exercise_session_id INTEGER,
                    score REAL,
                    time_of_appearance REAL,
                    FOREIGN KEY (exercise_session_id) REFERENCES exercise_sessions(exercise_session_id)
                )
            """
            )

            self.conn.commit()
            logging.info("Database tables created or verified successfully.")
        except Error as e:
            logging.error(f"Error creating tables: {e}")

    def add_time_of_appearance_column(self):
        """Add time_of_appearance column to pose_correctness and angle_correctness tables if they don't exist."""
        try:
            c = self.conn.cursor()
            # Check and add for pose_correctness
            c.execute("PRAGMA table_info(pose_correctness)")
            columns = [info[1] for info in c.fetchall()]
            if "time_of_appearance" not in columns:
                c.execute(
                    "ALTER TABLE pose_correctness RENAME COLUMN timestamp TO time_of_appearance;"
                )
                self.conn.commit()
                logging.info(
                    "Renamed 'timestamp' to 'time_of_appearance' in pose_correctness table."
                )
            else:
                logging.debug(
                    "'time_of_appearance' column already exists in pose_correctness table."
                )

            # Check and add for angle_correctness
            c.execute("PRAGMA table_info(angle_correctness)")
            columns = [info[1] for info in c.fetchall()]
            if "time_of_appearance" not in columns:
                c.execute(
                    "ALTER TABLE angle_correctness RENAME COLUMN timestamp TO time_of_appearance;"
                )
                self.conn.commit()
                logging.info(
                    "Renamed 'timestamp' to 'time_of_appearance' in angle_correctness table."
                )
            else:
                logging.debug(
                    "'time_of_appearance' column already exists in angle_correctness table."
                )

        except Error as e:
            logging.error(f"Error updating time_of_appearance columns: {e}")

    def rename_angle_correctness_table(self):
        """Rename angle_correctnmess to angle_correctness."""
        try:
            c = self.conn.cursor()
            # Check if 'angle_correctness' table exists
            c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='angle_correctness';"
            )
            if not c.fetchone():
                # Check if 'angle_correctnmess' table exists
                c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='angle_correctnmess';"
                )
                if c.fetchone():
                    c.execute(
                        "ALTER TABLE angle_correctnmess RENAME TO angle_correctness;"
                    )
                    self.conn.commit()
                    logging.info("Renamed table to 'angle_correctness'.")
                else:
                    logging.warning("'angle_correctnmess' table does not exist.")
            else:
                logging.debug("'angle_correctness' table already exists.")
        except Error as e:
            logging.error(f"Error renaming table: {e}")

    def insert_session(self, start_time, duration):
        """Insert a new session with start time and duration, and return its ID."""
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    INSERT INTO sessions (date, duration)
                    VALUES (?, ?)
                """,
                    (start_time.strftime("%Y-%m-%d %H:%M:%S"), duration),
                )
                session_id = cursor.lastrowid
                logging.info(f"Inserted new session with session_id {session_id}.")
                return session_id
        except Error as e:
            logging.error(f"Error inserting session: {e}")
            return None

    def insert_exercise_session(
        self,
        session_id,
        exercise,
        repetitions,
        duration,
        angle_correctness_count,
        pose_correctness_score,
    ):
        print(pose_correctness_score)
        """Insert an exercise record into the exercise_sessions table."""
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    INSERT INTO exercise_sessions (
                        session_id, exercise, repetitions, duration,
                        angle_correctness, pose_correctness_score
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        exercise,
                        repetitions,
                        duration,
                        angle_correctness_count,
                        (
                            round(pose_correctness_score, 2)
                            if pose_correctness_score is not None
                            else None
                        ),
                    ),
                )
                exercise_session_id = cursor.lastrowid
                logging.info(
                    f"Inserted exercise session with ID {exercise_session_id}."
                )
                return exercise_session_id
        except Error as e:
            logging.error(f"Error inserting exercise session: {e}")
            return None

    def insert_angle_correctness(
        self,
        exercise_session_id,
        angle_name,
        angle,
        expected_angle,
        threshold,
        comment,
        time_of_appearance,
    ):
        """Insert angle correctness feedback into the angle_correctness table with all relevant fields."""
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO angle_correctness (
                        exercise_session_id, angle_name, angle, expected_angle, threshold, comment, time_of_appearance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exercise_session_id,
                        angle_name,
                        angle,
                        expected_angle,
                        threshold,
                        comment,
                        time_of_appearance,
                    ),
                )
                logging.info(
                    f"Inserted angle correctness for exercise_session_id {exercise_session_id}."
                )
        except Error as e:
            logging.error(f"Error inserting angle correctness: {e}")

    def insert_pose_correctness(self, exercise_session_id, score, time_of_appearance):
        """Insert pose correctness data into the pose_correctness table."""
        try:
            with self.conn:
                rounded_score = round(score, 2) if score is not None else None
                self.conn.execute(
                    """
                    INSERT INTO pose_correctness (
                        exercise_session_id, score, time_of_appearance
                    ) VALUES (?, ?, ?)
                    """,
                    (exercise_session_id, rounded_score, time_of_appearance),
                )
                logging.info(
                    f"Inserted pose correctness for exercise_session_id {exercise_session_id}."
                )
        except Error as e:
            logging.error(f"Error inserting pose correctness: {e}")

    def update_session_duration(self, session_id, duration):
        """Update the duration of a session."""
        try:
            with self.conn:
                self.conn.execute(
                    """
                    UPDATE sessions
                    SET duration = ?
                    WHERE session_id = ?
                """,
                    (duration, session_id),
                )
                logging.info(
                    f"Updated duration for session_id {session_id} to {duration}."
                )
        except Error as e:
            logging.error(f"Error updating session duration: {e}")

    def update_angle_correctness_count(self, exercise_session_id, count):
        """Update the angle_correctness count in exercise_sessions."""
        try:
            with self.conn:
                self.conn.execute(
                    """
                    UPDATE exercise_sessions
                    SET angle_correctness = ?
                    WHERE exercise_session_id = ?
                """,
                    (count, exercise_session_id),
                )
                logging.info(
                    f"Updated angle_correctness count for exercise_session_id {exercise_session_id} to {count}."
                )
        except Error as e:
            logging.error(f"Error updating angle_correctness count: {e}")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")
