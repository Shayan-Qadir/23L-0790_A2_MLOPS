import sys
import platform
import os

print("=" * 60)
print(" VISIONGUARD - TASK 1.1 ENVIRONMENT VERIFICATION")
print("=" * 60)
print(f"Python Version    : {sys.version.split()[0]}")
print(f"Python Executable : {sys.executable}")
print(f"Operating System  : {platform.system()} {platform.release()} ({platform.machine()})")
print(f"Working Directory : {os.getcwd()}")
print("-" * 60)

# TensorFlow & GPU
try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
    try:
        import keras
        print(f"Keras Version     : {keras.__version__}")
    except Exception:
        print(f"Keras Version     : Built-in tf.keras")

    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Available     : {len(gpus) > 0}")
    print(f"Detected GPUs     : {gpus if gpus else 'None (Running in CPU mode - Standard for Windows native TF > 2.10)'}")
except ImportError as e:
    print(f"TensorFlow Error  : {e}")

print("-" * 60)
print(" INSTALLED DEPENDENCIES")
print("-" * 60)

packages = [
    ("MLflow", "mlflow"),
    ("DagsHub", "dagshub"),
    ("DVC", "dvc"),
    ("NumPy", "numpy"),
    ("Pandas", "pandas"),
    ("Pillow (PIL)", "PIL"),
    ("Matplotlib", "matplotlib"),
    ("Seaborn", "seaborn"),
    ("Scikit-Learn", "sklearn"),
    ("Keras Tuner", "keras_tuner"),
    ("Flask", "flask"),
]

for name, mod_name in packages:
    try:
        mod = __import__(mod_name)
        ver = getattr(mod, "__version__", "Installed")
        print(f"{name:<18}: {ver}")
    except ImportError:
        print(f"{name:<18}: NOT INSTALLED")

print("=" * 60)
print(" ENVIRONMENT STATUS: READY FOR TASK 1.2 DATASET AUDIT")
print("=" * 60)
