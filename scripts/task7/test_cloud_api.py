import os
import sys
import time
import json
from pathlib import Path
import requests

def test_api_endpoint(api_url="http://localhost:5000/api/predict", test_image_path=None, bypass_secret=None):
    """
    Sub-task 7 & Task 8 verification script:
    Sends image payload to VisionGuard API endpoint and prints response.
    Equivalent to Postman POST request testing.
    """
    # Normalize URL: ensure /api/predict is targeted if base domain given
    api_url = api_url.rstrip("/")
    if not api_url.endswith("/api/predict"):
        api_url += "/api/predict"

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
    
    headers = {}
    if bypass_secret:
        headers["x-vercel-protection-bypass"] = bypass_secret

    start_time = time.time()
    try:
        with open(test_image_path, "rb") as f:
            files = {"image": (os.path.basename(test_image_path), f, "image/jpeg")}
            response = requests.post(api_url, files=files, headers=headers, timeout=20)
            
        client_latency_ms = (time.time() - start_time) * 1000
        print(f"HTTP Status Code    : {response.status_code}")
        print(f"Roundtrip Latency   : {client_latency_ms:.1f} ms")
        print("\nResponse JSON Body:")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(f"[!] Response text (non-JSON): {response.text[:300]}")
        print("=" * 65)
        return response
    except Exception as e:
        print(f"[!] API Request Failed: {e}")
        print("=" * 65)
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VisionGuard API Test Harness")
    parser.add_argument("url", nargs="?", default="http://localhost:5000/api/predict", help="API URL")
    parser.add_argument("img", nargs="?", default=None, help="Image Path")
    parser.add_argument("--bypass-secret", "-s", default=os.getenv("VERCEL_BYPASS_SECRET"), help="Vercel Protection Bypass Secret")
    args = parser.parse_args()

    test_api_endpoint(api_url=args.url, test_image_path=args.img, bypass_secret=args.bypass_secret)

