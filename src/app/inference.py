import os
import io
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Dynamic import to support both heavy TensorFlow and lightweight TFLite (Vercel Serverless)
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter as TFLiteInterpreter
        except ImportError:
            TFLiteInterpreter = None

class VisionGuardInferenceEngine:
    """
    Singleton inference module that loads the champion model ONCE at startup.
    Supports both Keras SavedModel (.keras) and ultra-lean TensorFlow Lite (.tflite)
    for seamless local Docker and Vercel Serverless deployments.
    """
    _instance = None

    def __init__(self, model_path=None, threshold=0.50, model_version="Champion_MobileNetV2_v1"):
        if model_path is None:
            # Auto-detect TFLite on Vercel or if .keras is absent
            if os.getenv("VERCEL") or os.path.exists("models/champion/model.tflite"):
                default_path = "models/champion/model.tflite" if os.path.exists("models/champion/model.tflite") else "models/champion/model.keras"
            else:
                default_path = "models/champion/model.keras"
            model_path = os.getenv("MODEL_PATH", default_path)
            
        self.model_path = model_path
        self.threshold = float(os.getenv("CONFIDENCE_THRESHOLD", threshold))
        self.model_version = model_version
        self.model = None
        self.interpreter = None
        self.is_tflite = self.model_path.endswith(".tflite")
        self._load_model()

    @classmethod
    def get_instance(cls, model_path=None, threshold=0.50):
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, threshold=threshold)
        return cls._instance

    def _load_model(self):
        print(f"[*] Initializing VisionGuard Inference Engine...")
        if not os.path.exists(self.model_path):
            # Fallback if preferred format not found
            alt_path = "models/champion/model.tflite" if self.model_path.endswith(".keras") else "models/champion/model.keras"
            if os.path.exists(alt_path):
                self.model_path = alt_path
                self.is_tflite = alt_path.endswith(".tflite")
            else:
                print(f"[!] Warning: Model file not found at {self.model_path}. Engine will operate once model is trained.")
                return
            
        start_time = time.time()
        if self.is_tflite:
            if TFLiteInterpreter is None:
                raise RuntimeError("No TFLite interpreter available (ai_edge_litert, tflite_runtime, or tensorflow.lite).")
            self.interpreter = TFLiteInterpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            load_duration = (time.time() - start_time) * 1000
            print(f"[+] Champion TFLite model loaded successfully in {load_duration:.1f}ms from: {self.model_path}")
        else:
            if tf is None:
                raise RuntimeError("TensorFlow is required to load .keras model.")
            self.model = tf.keras.models.load_model(self.model_path)
            load_duration = (time.time() - start_time) * 1000
            print(f"[+] Champion Keras model loaded successfully in {load_duration:.1f}ms from: {self.model_path}")

    def preprocess_image(self, image_bytes, target_size=(224, 224)):
        """
        Preprocesses uploaded image in-memory without saving to disk.
        Converts to 3-channel RGB, resizes to target_size, and returns float32 tensor.
        """
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
            img_array = np.array(pil_img, dtype=np.float32)

        # Expand batch dimension: (1, 224, 224, 3)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_bytes, filename="uploaded_image"):
        """
        Runs inference on uploaded image bytes.
        Returns: decision label, confidence percentage, raw probability, and latency.
        """
        if self.model is None and self.interpreter is None:
            self._load_model()
            if self.model is None and self.interpreter is None:
                raise RuntimeError("Champion model is not loaded. Please train or provide model weights.")

        start_time = time.time()
        input_tensor = self.preprocess_image(image_bytes)
        
        # Forward pass
        if self.is_tflite:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            self.interpreter.invoke()
            raw_prob = float(self.interpreter.get_tensor(self.output_details[0]["index"])[0, 0])
        else:
            raw_prob = float(self.model.predict(input_tensor, verbose=0)[0, 0])
            
        latency_ms = (time.time() - start_time) * 1000

        # Decision & Confidence calibration
        is_ai = raw_prob >= self.threshold
        if is_ai:
            decision = "AI-Generated"
            confidence = raw_prob
        else:
            decision = "Real Artwork"
            confidence = 1.0 - raw_prob

        return {
            "filename": filename,
            "decision": decision,
            "is_ai": bool(is_ai),
            "confidence": float(confidence),
            "confidence_percentage": round(confidence * 100, 2),
            "raw_probability": float(raw_prob),
            "threshold": self.threshold,
            "model_version": self.model_version,
            "latency_ms": round(latency_ms, 2)
        }

def test_inference_on_known_images():
    """Sub-task 4.1 verification: Test two known images (1 Real, 1 AI)."""
    print("=" * 65)
    print(" TASK 4.1: INFERENCE ENGINE VERIFICATION ON KNOWN IMAGES")
    print("=" * 65)
    
    engine = VisionGuardInferenceEngine.get_instance()
    
    # Locate one AI image and one Real image from data/raw
    ai_dir = Path("data/raw/AiArtData")
    real_dir = Path("data/raw/RealArt")
    
    ai_sample = next(ai_dir.glob("*.jpg"), None) or next(ai_dir.glob("*.png"), None)
    real_sample = next(real_dir.glob("*.jpg"), None) or next(real_dir.glob("*.png"), None)
    
    for path, expected in [(real_sample, "Real Artwork"), (ai_sample, "AI-Generated")]:
        if path and path.exists():
            with open(path, "rb") as f:
                img_bytes = f.read()
            res = engine.predict(img_bytes, filename=path.name)
            print(f"\n[TESTING]: {path.name}")
            print(f"  Expected Class : {expected}")
            print(f"  Model Verdict  : {res['decision']} (Raw Prob: {res['raw_probability']:.4f})")
            print(f"  Confidence     : {res['confidence_percentage']}%")
            print(f"  Inference Time : {res['latency_ms']} ms")
            print(f"  Matches Expected: {res['decision'] == expected}")
    print("=" * 65)

if __name__ == "__main__":
    test_inference_on_known_images()

