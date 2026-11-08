import os
import io
import time
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

class VisionGuardInferenceEngine:
    """
    Singleton inference module that loads the champion Keras model ONCE at startup.
    
    Technical Justification:
    1. Why loading per request is unacceptable:
       - Instantiating a deep learning model, compiling computational graph kernels,
         and loading hundreds of megabytes of weight tensors from disk into RAM on every
         HTTP request introduces massive I/O latency (often 1000ms - 5000ms per request).
       - Under concurrent web traffic, multiple simultaneous disk-load calls exhaust
         server memory and cause CPU thrashing or out-of-memory crashes.
       - Loading once at startup retains the compiled computation graph in memory,
         reducing per-request inference time to a few milliseconds.
         
    2. Why training and serving preprocessing must match:
       - Deep neural networks learn decision boundaries conditioned strictly on the
         numerical input distribution encountered during training.
       - If training used MobileNetV2 preprocessing (scaling pixel values [0, 255] -> [-1, 1])
         and bilinear interpolation to 224x224, serving with [0, 1] scaling or different
         channel ordering produces a massive covariate shift, causing the network to emit
         erratic, degraded confidence scores.
    """
    _instance = None

    def __init__(self, model_path=None, threshold=0.50, model_version="Champion_MobileNetV2_v1"):
        if model_path is None:
            model_path = os.getenv("MODEL_PATH", "models/champion/model.keras")
        self.model_path = model_path
        self.threshold = float(os.getenv("CONFIDENCE_THRESHOLD", threshold))
        self.model_version = model_version
        self.model = None
        self._load_model()

    @classmethod
    def get_instance(cls, model_path=None, threshold=0.50):
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, threshold=threshold)
        return cls._instance

    def _load_model(self):
        print(f"[*] Initializing VisionGuard Inference Engine...")
        if not os.path.exists(self.model_path):
            print(f"[!] Warning: Model file not found at {self.model_path}. Engine will operate once model is trained.")
            return
            
        start_time = time.time()
        # Load model weights into memory once
        self.model = tf.keras.models.load_model(self.model_path)
        load_duration = (time.time() - start_time) * 1000
        print(f"[+] Champion model loaded successfully into memory in {load_duration:.1f}ms from: {self.model_path}")

    def preprocess_image(self, image_bytes, target_size=(224, 224)):
        """
        Preprocesses uploaded image in-memory without saving to disk.
        Converts to 3-channel RGB, resizes to target_size, and returns float32 tensor.
        """
        # Read image in-memory via BytesIO
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            # Ensure standard RGB mode (strips transparency/palette)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            # Resize
            pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
            img_array = np.array(pil_img, dtype=np.float32)

        # Expand batch dimension: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predict(self, image_bytes, filename="uploaded_image"):
        """
        Runs inference on uploaded image bytes.
        Returns: decision label, confidence percentage, raw probability, and latency.
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise RuntimeError("Champion model is not loaded. Please train or provide model weights.")

        start_time = time.time()
        
        # Preprocessing in-memory
        input_tensor = self.preprocess_image(image_bytes)
        
        # Forward pass
        raw_prob = float(self.model.predict(input_tensor, verbose=0)[0, 0])
        latency_ms = (time.time() - start_time) * 1000

        # Decision & Confidence calibration
        is_ai = raw_prob >= self.threshold
        if is_ai:
            decision = "AI-Generated"
            confidence = raw_prob  # Degree of certainty for AI
        else:
            decision = "Real Artwork"
            confidence = 1.0 - raw_prob  # Degree of certainty for Real

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

