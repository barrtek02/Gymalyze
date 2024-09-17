import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Path to the database file
DATABASE_PATH = os.path.join(BASE_DIR, "data/app_data.db")
