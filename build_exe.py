import os
import subprocess

# Specify the relative path to your main application script
script_name = "src/main.py"  # Adjust this if your entry point differs

# Check if the script exists
if not os.path.exists(script_name):
    print(f"Error: {script_name} does not exist")
else:
    print("Starting the build process...")

    # Run pyinstaller to create the executable
    result = subprocess.run(
        [
            "pyinstaller",
            "--noconsole",  # Hide console (useful for GUI apps)
            "--name=Gymalyze",  # Name of the generated EXE
            script_name,  # The main script to compile
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Ensures output is returned as strings
    )

    # Check if pyinstaller ran successfully
    if result.returncode != 0:
        print(f"Error: pyinstaller failed with code {result.returncode}")
        print(f"Output:\n{result.stdout}")
        print(f"Error:\n{result.stderr}")
    else:
        print("Executable created successfully in the 'dist' directory")
