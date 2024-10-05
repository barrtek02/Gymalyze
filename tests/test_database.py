import sys
import os

from src.utils.database import Database

# Add src/ to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest


class TestDatabase(unittest.TestCase):

    def setUp(self):
        """Set up a temporary in-memory database for testing."""
        self.db = Database(":memory:")  # Use an in-memory database for tests
        self.db.create_table()  # Ensure tables are created

    def test_insert_user(self):
        """Test that a user can be inserted into the database."""
        self.db.insert_user("testuser", "testpassword")
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", ("testuser",))
        user = cursor.fetchone()
        self.assertIsNotNone(user, "User was not inserted into the database")
        self.assertEqual(user[1], "testuser", "Username does not match")
        self.assertEqual(user[2], "testpassword", "Password does not match")

    def test_validate_user(self):
        """Test user validation works correctly."""
        self.db.insert_user("validuser", "validpassword")
        valid_user = self.db.validate_user("validuser", "validpassword")
        invalid_user = self.db.validate_user("invaliduser", "wrongpassword")

        self.assertIsNotNone(valid_user, "Valid user should be found")
        self.assertIsNone(invalid_user, "Invalid user should not be found")

    def tearDown(self):
        """Close the database after tests."""
        self.db.close()


if __name__ == "__main__":
    unittest.main()
