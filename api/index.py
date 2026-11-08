import os
import sys

# Ensure root directory is on the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# If model.tflite exists, set MODEL_PATH to it
tflite_path = os.path.join(root_dir, "models", "champion", "model.tflite")
if os.path.exists(tflite_path):
    os.environ.setdefault("MODEL_PATH", tflite_path)

from src.app.app import app

# Vercel entrypoint
app.debug = False

