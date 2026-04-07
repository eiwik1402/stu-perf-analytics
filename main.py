import subprocess
import os

def main():
    gui_path = os.path.join(os.path.dirname(__file__), "gui", "gui.py")
    subprocess.run(["streamlit", "run", gui_path])

if __name__ == "__main__":
    main()