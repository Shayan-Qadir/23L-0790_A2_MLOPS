import os
import sys
import time
import json
from pathlib import Path
import requests

def test_api_endpoint(api_url="http://localhost:5000/api/predict", test_image_path=None):
    """
    Sub-task 7 & Task 8 verification script:
    Sends image payload to VisionGuard API endpoint and prints response.
    Equivalent to Postman POST request testing.
    """
    print("=" * 65)
    print(" VISIONGUARD API TEST HARNESS (Task 7 & Task 8)")
    print("=" * 65)
    print(f"Target API Endpoint : {api_url}")
    
    # Select test image if not provided
    if not test_image_path or not os.path.exists(test_image_path):
        ai_sample = next(Path("data/raw/AiArtData").glob("*.jpg"), None)
        test_image_path = str(ai_sample) if ai_sample else "tests/sample.jpg"
        
    print(f"Test Image Selected : {test_image_path}")
    print("-" * 65)
    
    start_time = time.time()
    try:
        with open(test_image_path, "rb") as f:
            files = {"image": (os.path.basename(test_image_path), f, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=15)
            
        client_latency_ms = (time.time() - start_time) * 1000
        print(f"HTTP Status Code    : {response.status_code}")
        print(f"Roundtrip Latency   : {client_latency_ms:.1f} ms")
        print("\nResponse JSON Body:")
        print(json.dumps(response.json(), indent=2))
        print("=" * 65)
        return response.json()
    except Exception as e:
        print(f"[!] API Request Failed: {e}")
        print("=" * 65)
        return None

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000/api/predict"
    img = sys.argv[2] if len(sys.argv) > 2 else None
    test_api_endpoint(api_url=url, test_image_path=img)

