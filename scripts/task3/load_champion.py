import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from src.models.pipeline import parse_image_and_label

def setup_model_registry(run_id_v1="452bfbf439344790abe6c53ea34477c4", run_id_v2="427482b32fa54d80b7d396859a25774e", model_name="VisionGuard"):
    """
    Sub-task 3.6:
    Registers model in DagsHub MLflow Model Registry.
    Creates at least two versions and assigns the alias 'champion' to the best one.
    """
    load_dotenv(".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    print("=" * 65)
    print(" TASK 3.6: MLFLOW MODEL REGISTRY CONFIGURATION")
    print("=" * 65)
    print(f"Tracking URI : {tracking_uri}")
    print(f"Model Name   : {model_name}")
    
    # Ensure registered model exists
    try:
        client.create_registered_model(model_name, description="VisionGuard AI Art Detector Registry")
        print(f"[+] Created registered model: {model_name}")
    except Exception:
        print(f"[*] Registered model '{model_name}' already exists.")
        
    print(f"[*] Registering Version 1 from Run: {run_id_v1[:8]}...")
    try:
        mv1 = client.create_model_version(name=model_name, source=f"runs:/{run_id_v1}/model", run_id=run_id_v1)
        print(f"[+] Version 1 registered: {mv1.name} (v{mv1.version})")
    except Exception as e:
        print(f"[*] Version 1 status: {e}")
        
    print(f"[*] Registering Version 2 from Run: {run_id_v2[:8]}...")
    try:
        mv2 = client.create_model_version(name=model_name, source=f"runs:/{run_id_v2}/model", run_id=run_id_v2)
        print(f"[+] Version 2 registered: {mv2.name} (v{mv2.version})")
    except Exception as e:
        print(f"[*] Version 2 status: {e}")
        
    # Assign alias 'champion' to Version 1 (the Fine-Tuned MobileNetV2 with best AUC and accuracy)
    champion_version = "1"
    print(f"[*] Assigning alias '@champion' to Version {champion_version}...")
    try:
        client.set_registered_model_alias(model_name, "champion", champion_version)
        print(f"[+] Alias '@champion' successfully assigned to {model_name} version {champion_version}!")
    except Exception as e:
        print(f"[!] Alias assignment: {e}")
        
    # Verify alias resolution
    model_version_champion = client.get_model_version_by_alias(model_name, "champion")
    print(f"[+] Verified alias '@champion' points to: Version {model_version_champion.version} (Run: {model_version_champion.run_id})")
    print("=" * 65)

def load_champion_and_predict(model_name="VisionGuard", test_image_path=None):
    """
    Sub-task 3.6 script:
    Loads the model by alias '@champion' and classifies one image.
    Evidence for report: Terminal output of the load and prediction.
    """
    load_dotenv(".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    
    print("\n" + "=" * 65)
    print(" TASK 3.6: LOAD MODEL BY ALIAS AND CLASSIFY IMAGE")
    print("=" * 65)
    
    champion_uri = f"models:/{model_name}@champion"
    print(f"[*] Resolving Model URI: {champion_uri} from MLflow...")
    
    start_time = tf.timestamp()
    try:
        model = mlflow.tensorflow.load_model(champion_uri)
        print(f"[+] Successfully loaded champion model from registry!")
    except Exception as e:
        print(f"[!] Registry load fallback to local champion weights: {e}")
        model = tf.keras.models.load_model("models/champion/model.keras")
        print("[+] Loaded from models/champion/model.keras")
        
    if not test_image_path or not os.path.exists(test_image_path):
        sample = next(Path("data/raw/AiArtData").glob("*.jpg"), None)
        test_image_path = str(sample) if sample else "tests/sample.jpg"
        
    print(f"[*] Classifying image: {test_image_path}")
    img, _ = parse_image_and_label(test_image_path, 0)
    img_batch = tf.expand_dims(img, 0)
    
    prob = float(model.predict(img_batch, verbose=0)[0, 0])
    is_ai = prob >= 0.50
    verdict = "AI-Generated Artwork" if is_ai else "Authentic Human Artwork"
    confidence = prob if is_ai else (1.0 - prob)
    
    print("-" * 65)
    print(f" PREDICTION RESULT:")
    print(f"  Target File : {os.path.basename(test_image_path)}")
    print(f"  Verdict     : {verdict}")
    print(f"  Probability : {prob:.4f}")
    print(f"  Confidence  : {confidence * 100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    load_champion_and_predict(test_image_path=test_img)

