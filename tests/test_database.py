import pytest
import os
from datetime import datetime
from src.utils.database import Database

@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture to provide a temporary database path."""
    return str(tmp_path / "test_app_data.db")

@pytest.fixture
def db(temp_db_path):
    """Fixture to initialize the Database object with a temporary database."""
    database = Database(db_path=temp_db_path)
    yield database
    database.close()

def test_create_connection(db):
    """Test that the database connection is successfully created."""
    assert db.conn is not None
    assert os.path.exists(db.db_path)

def test_create_tables(db):
    """Test that tables are created successfully."""
    conn = db.conn
    cursor = conn.cursor()

    # Check for the existence of tables
    tables = ["sessions", "exercise_sessions", "angle_correctness", "pose_correctness"]
    for table in tables:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        assert cursor.fetchone() is not None

def test_insert_session(db):
    """Test inserting a session into the database."""
    start_time = datetime.now()
    duration = 60.5

    session_id = db.insert_session(start_time, duration)
    assert session_id is not None

    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    session = cursor.fetchone()

    assert session is not None
    assert session[1] == start_time.strftime("%Y-%m-%d %H:%M:%S")
    assert session[2] == duration

def test_insert_exercise_session(db):
    """Test inserting an exercise session into the database."""
    session_id = db.insert_session(datetime.now(), 60.5)
    exercise = "Push-up"
    repetitions = 15
    duration = 30.0
    angle_correctness_count = 5
    pose_correctness_score = 87.5

    exercise_session_id = db.insert_exercise_session(
        session_id, exercise, repetitions, duration, angle_correctness_count, pose_correctness_score
    )
    assert exercise_session_id is not None

    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT * FROM exercise_sessions WHERE exercise_session_id=?", (exercise_session_id,)
    )
    exercise_session = cursor.fetchone()

    assert exercise_session is not None
    assert exercise_session[1] == session_id
    assert exercise_session[2] == exercise
    assert exercise_session[3] == repetitions
    assert exercise_session[4] == duration
    assert exercise_session[5] == angle_correctness_count
    assert exercise_session[6] == round(pose_correctness_score, 2)

def test_insert_angle_correctness(db):
    """Test inserting angle correctness feedback."""
    session_id = db.insert_session(datetime.now(), 60.5)
    exercise_session_id = db.insert_exercise_session(session_id, "Squats", 10, 40.0, 3, 95.0)

    db.insert_angle_correctness(
        exercise_session_id,
        "Knee Angle",
        85.0,
        90.0,
        5.0,
        "Good form",
        15.0,
    )

    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT * FROM angle_correctness WHERE exercise_session_id=?", (exercise_session_id,)
    )
    angle_correctness = cursor.fetchone()

    assert angle_correctness is not None
    assert angle_correctness[1] == exercise_session_id
    assert angle_correctness[2] == "Knee Angle"
