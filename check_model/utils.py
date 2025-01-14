import os
from pathlib import Path

PROJECT_DIR = os.getcwd()
# Get the current directory with pathlib
PROJECT_DIR = Path(PROJECT_DIR).resolve().parents[0]

INPUT_TEST_DIR = os.path.join(PROJECT_DIR, 'datasets', 'input-test')
DATABASE_DIR = os.path.join(PROJECT_DIR, 'database')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'settings', 'settings.cfg')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')


