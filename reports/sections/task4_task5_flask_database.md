# Tasks 4 & 5: Flask Web Application and Database Trail

## 4.1 Inference Engine Architecture
Implemented in `src/app/inference.py`:
* **Singleton Lifecycle Pattern:** The champion model is instantiated and retained in memory exactly once when the server initializes (`VisionGuardInferenceEngine.get_instance()`).
* **Why Loading Per Request is Unacceptable:**
  * Re-instantiating a deep learning model on every HTTP request requires reading tens of megabytes of serialized weights from disk into RAM, parsing computation graph definitions, and compiling execution kernels.
  * This creates severe response latency ($\ge 1000 - 3000$ ms per transaction).
  * Under concurrent web traffic, multiple simultaneous load operations lead to memory bloat, high CPU usage, and server crashes.
  * Loading once at startup reduces per-request inference latency to $< 40$ ms.
* **Why Training and Serving Preprocessing Must Match:**
  * Neural networks are parametric function approximators whose weights are calibrated against the exact numerical distribution of their training inputs.
  * Training MobileNetV2 uses `preprocess_input` (scaling pixel intensities from $[0, 255]$ to $[-1.0, 1.0]$) and resizing to $224 \times 224$ with bilinear interpolation.
  * If the serving layer applied standard $[0, 1]$ rescaling or different color channel ordering, it would introduce a severe **covariate shift**, causing degraded and erratic predictions.

---

## 4.2 Routes, Security & Input Validation
Implemented in `src/app/app.py`:
1. **Sanitization:** All uploaded filenames are stripped of path traversal vectors (`../`, absolute paths) using `werkzeug.utils.secure_filename`.
2. **Strict Extension Whitelist:** Only `.jpg`, `.jpeg`, `.png`, and `.webp` are accepted. Files with unapproved extensions trigger a friendly `400 Bad Request` page.
3. **Payload Size Guard:** Enforces `MAX_CONTENT_LENGTH = 5.0 MB`. Oversized files trigger a custom `413 Payload Too Large` error page.
4. **Integrity Verification:** Inspects image byte headers in-memory via `PIL.Image.verify()`. Corrupted files, hollow payloads, or renamed executables are rejected with a descriptive `400 Bad Request`.
5. **Zero Disk Storage:** User files are processed strictly via memory buffers (`io.BytesIO`). No temporary files are saved to the server filesystem, ensuring zero disk clutter and protecting user data privacy.

---

## 4.4 Configuration & Production Hardening
* All critical runtime settings are decoupled into environment variables (`.env`):
  * `MODEL_PATH`: Location of the serialized champion Keras model.
  * `CONFIDENCE_THRESHOLD`: Decision boundary for class assignment.
  * `DATABASE_URL`: Connection string for PostgreSQL or SQLite.
  * `FLASK_DEBUG=0`: Production mode with debugging disabled.
* **Why Debug Mode Must Be Off in Production:**
  * Werkzeug's interactive web debugger allows arbitrary Python code execution directly in the browser upon uncaught exceptions.
  * Debug mode exposes server file paths, environment variables, and stack traces to external visitors, creating serious security vulnerabilities.
  * It also enables hot-reloading watchers that increase CPU overhead.

---

## Task 5: Database Trail
Implemented in `src/app/database.py`:
* **Schema Design (`prediction_logs`):**
  * `id`: Auto-incrementing primary key.
  * `timestamp`: UTC datetime of the request.
  * `filename`: Sanitized original image name.
  * `decision`: Qualitative classification (`AI-Generated` vs `Real Artwork`).
  * `confidence`: Scaled percentage confidence ($0.0 - 1.0$).
  * `raw_probability`: Raw sigmoid output from the neural network.
  * `threshold_used`: The decision threshold active at inference time.
  * `model_version`: Tag/alias of the active model (`Champion_MobileNetV2_v1`).
  * `inference_time_ms`: Processing latency in milliseconds.
* **UI Integration:**
  * Every inference transaction inserts a new row into the database.
  * The `/history` view renders an audit table displaying recent searches, verdicts, confidence metrics, and timestamps.

