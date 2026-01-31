import os
import sys
import subprocess
import venv

def create_venv(venv_path='venv'):
    """Create a virtual environment at the given path."""
    if not os.path.exists(venv_path):
        print(f"Creating virtual environment at {venv_path}...")
        venv.create(venv_path, with_pip=True)
    else:
        print(f"Virtual environment already exists at {venv_path}")

def install_requirements(venv_path='venv', requirements_file='requirements.txt'):
    """Install packages from requirements.txt into the venv."""
    # Determine the path to the python executable inside the venv
    if os.name == 'nt':  # Windows
        python_executable = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:  # macOS/Linux
        python_executable = os.path.join(venv_path, 'bin', 'python')

    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"Python executable not found in {python_executable}")

    # Install requirements
    if os.path.exists(requirements_file):
        print(f"Installing requirements from {requirements_file}...")
        subprocess.check_call([python_executable, '-m', 'pip', 'install', '-r', requirements_file])
    else:
        print(f"No requirements file found at {requirements_file}, skipping installation.")

if __name__ == "__main__":
    venv_path = 'venv'  # change if you want a different name
    requirements_file = 'requirements.txt'

    create_venv(venv_path)
    install_requirements(venv_path, requirements_file)