import os
import sys
import io
import base64
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image

# Ensure local imports
sys.path.insert(0, os.path.abspath("."))
from src.app.database import db, init_db, log_prediction, get_recent_history
from src.app.inference import VisionGuardInferenceEngine

# Load environment configuration
load_dotenv(".env")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "visionguard-secret-key-23L0790")

# Database Configuration
db_url = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
os.makedirs(app.instance_path, exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Maximum payload size: 5.0 Megabytes (Sub-task 4.2)
MAX_FILE_SIZE_MB = 5.0
app.config["MAX_CONTENT_LENGTH"] = int(MAX_FILE_SIZE_MB * 1024 * 1024)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Initialize Database
init_db(app)

# Initialize Inference Engine Singleton (loaded once at startup)
inference_engine = VisionGuardInferenceEngine.get_instance(
    model_path=os.getenv("MODEL_PATH", "models/champion/model.keras"),
    threshold=float(os.getenv("CONFIDENCE_THRESHOLD", 0.50))
)

def allowed_file(filename):
    """Verifies that filename has an approved image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------------
# ERROR HANDLERS (Friendly Error Pages for Sub-task 4.2)
# -------------------------------------------------------------
@app.errorhandler(413)
def file_too_large(e):
    return render_template(
        "error.html",
        error_code="413",
        error_title="Payload Exceeds Size Limit",
        error_message=f"The selected file exceeds the maximum permitted limit of {MAX_FILE_SIZE_MB} MB.",
        error_guidance="Please compress your image or select an optimized image file (< 5.0 MB) and try again."
    ), 413

@app.errorhandler(400)
def bad_request(e):
    return render_template(
        "error.html",
        error_code="400",
        error_title="Invalid Image Request",
        error_message=str(e.description if hasattr(e, "description") else "The submitted request was malformed or could not be decoded."),
        error_guidance="Verify that the file is an uncorrupted JPEG, PNG, or WEBP image."
    ), 400

@app.errorhandler(404)
def not_found(e):
    return render_template(
        "error.html",
        error_code="404",
        error_title="Page Not Found",
        error_message="The requested endpoint does not exist.",
        error_guidance="Navigate back to the home page to inspect images."
    ), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template(
        "error.html",
        error_code="500",
        error_title="Inference Service Exception",
        error_message="An internal error occurred while processing model inference.",
        error_guidance="The error has been logged. Verify that model weights are present."
    ), 500

# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Home upload page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Sub-task 4.2: Handles image upload, in-memory validation,
    sanitization, inference, and database logging without saving to disk.
    """
    if "image" not in request.files:
        flash("No file part in the submission.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No image file was selected.", "error")
        return redirect(url_for("index"))

    # 1. Sanitize filename
    clean_filename = secure_filename(file.filename)
    if not clean_filename:
        clean_filename = "unnamed_image.jpg"

    # 2. Reject disallowed file extension
    if not allowed_file(clean_filename):
        return render_template(
            "error.html",
            error_code="400",
            error_title="Unsupported File Extension",
            error_message=f"The file '{clean_filename}' does not have a permitted image extension.",
            error_guidance="Supported file extensions are: .jpg, .jpeg, .png, and .webp."
        ), 400

    # 3. Read bytes in-memory (Zero disk persistence)
    try:
        image_bytes = file.read()
    except Exception as e:
        return render_template(
            "error.html",
            error_code="400",
            error_title="File Stream Error",
            error_message="Unable to read the uploaded byte stream.",
            error_guidance=str(e)
        ), 400

    # 4. Reject corrupt files or invalid image contents
    try:
        with Image.open(io.BytesIO(image_bytes)) as img_test:
            img_test.verify()
    except Exception as e:
        return render_template(
            "error.html",
            error_code="400",
            error_title="Corrupted or Unreadable Image",
            error_message=f"The file '{clean_filename}' contains invalid or corrupted image data and could not be decoded.",
            error_guidance="Ensure the file is a valid raster image file and not an empty or renamed non-image format."
        ), 400

    # 5. Check if remote API is enabled (Task 8 integration)
    use_remote_api = os.getenv("USE_REMOTE_API", "0").lower() in ("1", "true")
    if use_remote_api:
        import requests as http_requests
        remote_url = os.getenv("REMOTE_API_URL", "http://localhost:5000/api/predict")
        try:
            resp = http_requests.post(
                remote_url,
                files={"image": (clean_filename, io.BytesIO(image_bytes), "image/jpeg")},
                timeout=10
            )
            result = resp.json()
        except Exception as e:
            flash(f"Remote API call failed: {e}. Falling back to local inference.", "warning")
            result = inference_engine.predict(image_bytes, filename=clean_filename)
    else:
        # Standard local inference
        result = inference_engine.predict(image_bytes, filename=clean_filename)

    # 6. Task 5: Persist permanent record into database
    log_entry = log_prediction(
        filename=clean_filename,
        decision=result["decision"],
        confidence=result["confidence"],
        raw_probability=result["raw_probability"],
        threshold_used=result["threshold"],
        model_version=result["model_version"],
        inference_time_ms=result["latency_ms"]
    )

    # 7. Convert image bytes to Base64 for web display (without saving on disk)
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    return render_template(
        "result.html",
        filename=clean_filename,
        decision=result["decision"],
        is_ai=result["is_ai"],
        confidence_percentage=result["confidence_percentage"],
        raw_probability=result["raw_probability"],
        threshold=result["threshold"],
        latency_ms=result["latency_ms"],
        model_version=result["model_version"],
        image_base64=img_b64,
        log_id=log_entry.id
    )

@app.route("/history", methods=["GET"])
def history():
    """Task 5: Display past search history from database."""
    logs = get_recent_history(limit=50)
    return render_template("history.html", logs=logs)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Docker Compose & Cloud monitoring."""
    is_loaded = inference_engine.model is not None
    return jsonify({
        "status": "healthy" if is_loaded else "degraded",
        "service": "VisionGuard Web App",
        "model_loaded": is_loaded,
        "model_version": inference_engine.model_version,
        "threshold": inference_engine.threshold,
        "timestamp": os.getenv("CURRENT_TIME", "")
    }), 200 if is_loaded else 503

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for Cloud API deployment & Task 7/8 calls."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    image_bytes = file.read()
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception:
        return jsonify({"error": "Invalid or corrupt image format"}), 400

    result = inference_engine.predict(image_bytes, filename=secure_filename(file.filename))
    return jsonify(result), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # Run server with debug mode OFF per specification 4.4
    app.run(host="0.0.0.0", port=port, debug=False)

