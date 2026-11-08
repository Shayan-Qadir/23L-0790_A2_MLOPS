# Task 1: Data Understanding and Versioning

## 1.1 Environment Verification
* **Python Runtime:** Python 3.13.5 (64-bit) running within an isolated virtual environment (`.venv`).
* **Deep Learning Framework:** TensorFlow 2.21.0, Keras 3.15.1.
* **MLOps & Tracking:** MLflow 3.18.0, DagsHub 0.7.3, DVC 3.67.1.
* **Data & Scientific Libraries:** NumPy 2.5.4, Pandas 3.1.1, Pillow 13.0.0, Matplotlib 3.11.2, Seaborn 0.13.2, Scikit-Learn 1.9.1, Keras Tuner 1.4.8, Flask 3.1.3.
* **GPU Detection:** 
  `tf.config.list_physical_devices('GPU')` returned `[]` (CPU execution mode).
  *Technical Rationale:* On native Windows operating systems, TensorFlow discontinued direct native GPU support for releases beyond TF 2.10. GPU acceleration on Windows is formally supported via the Windows Subsystem for Linux (WSL2) or the TensorFlow-DirectML plugin. CPU execution is fully sufficient for our training and inference requirements, and containerization in Task 6 utilizes `tensorflow-cpu` to reduce image size.

---

## 1.2 Dataset Audit
A comprehensive audit script (`src/data/dataset_audit.py`) scanned all image files across the dataset directory (`data/raw`). Each file was verified for byte integrity, extracted for spatial dimensions and color encoding, and hashed via SHA-256 for duplicate detection.

### Summary Statistics
| Metric | Overall | AiArtData (AI-Generated) | RealArt (Authentic) |
| :--- | :--- | :--- | :--- |
| **Total Images** | **975** (100.0%) | **539** (55.28%) | **436** (44.72%) |
| **Corrupt / Unreadable Files** | **0** | 0 | 0 |
| **Exact Duplicate Files** | **2** | 2 | 0 |
| **Width (Min / Median / Max)** | 183 / 1024 / 8495 px | 225 / 1024 / 6000 px | 183 / 920 / 8495 px |
| **Height (Min / Median / Max)** | 135 / 768 / 7091 px | 135 / 800 / 4001 px | 158 / 720 / 7091 px |
| **File Size (Min / Median / Max)** | 11.0 / 165.1 / 9956.4 KB | 11.5 / 195.1 / 5494.3 KB | 11.0 / 141.4 / 9956.4 KB |

### Encoding & Formats
* **Color Modes:**
  * `AiArtData`: 489 RGB, 35 RGBA, 14 Palette (P), 1 Grayscale (L).
  * `RealArt`: 407 RGB, 14 RGBA, 9 Palette (P), 6 Grayscale (L).
* **File Formats:**
  * `AiArtData`: 429 JPEG, 110 PNG.
  * `RealArt`: 403 JPEG, 33 PNG.
* **Duplicate Finding:**
  Exact duplicates identified by SHA-256 hash `f3268a4fc0477b2f3d9728f118afe32bf5196c3b476def53e34229672f77dc97`:
  `image33.jpg` and `image34.jpg` in `AiArtData`.

---

## 1.3 Visual Exploratory Data Analysis (EDA)
Generated high-resolution visualizations are preserved in `reports/figures/eda/`:
1. `sample_grid.png`: Visual comparison of representative samples from both classes with spatial and byte dimensions.
2. `histograms_dimensions.png`: Comparative histograms and kernel density estimations of image widths and heights.
3. `histograms_filesize.png`: Linear and logarithmic distributions of file storage footprints.
4. `aspect_ratio_and_formats.png`: Box plots of aspect ratios and bar charts of JPEG vs PNG encodings.

*(Full captions and 2-3 sentence interpretations are provided in `reports/figures/eda/eda_interpretations.md`.)*

---

## 1.4 Data Version Control (DVC)

### Toy Demonstration
Executed via `scripts/task1/run_toy_dvc.py`:
1. Initialized Git repository and DVC workspace (`dvc init`).
2. Tracked sample dataset (`sample_data.csv`) under DVC, generating pointer file `sample_data.csv.dvc`.
3. Added and pushed Version 1 to local DVC remote storage (`toy_remote`).
4. Updated file with additional rows (Version 2), committed, and pushed to remote.
5. Successfully checked out previous Git commit (`git checkout HEAD~1 sample_data.csv.dvc`) and restored Version 1 content via `dvc checkout`.
6. Verified remote recovery: deleted local `sample_data.csv` and restored it flawlessly using `dvc pull`.

### Project Application
* Dataset directory `data/raw` is tracked by DVC via pointer file `data/raw.dvc`.
* Remote configured to DagsHub Storage: `https://dagshub.com/Shayan-Qadir/23L0790_A2_Ops.dvc`.
* Pointer file committed to Git; binary images uploaded to DagsHub DVC remote.

### Technical Explanation: The Small Pointer File & Git Storage Limitations
1. **What the small pointer file holds:**
   The `.dvc` file is a lightweight YAML metadata file containing:
   * `md5`: A content-addressable hash uniquely representing the directory structure and file contents (e.g., `8f23d6dc9b06d363c3981642c50856c0.dir`).
   * `size`: The total cumulative byte size of the tracked payload (503,783,378 bytes $\approx$ 503.8 MB).
   * `nfiles`: The exact number of tracked files (975).
   * `path`: The local relative workspace destination (`raw`).
2. **Why large data does not belong in Git:**
   * **Lack of Delta Compression for Binaries:** Git utilizes directed acyclic graph (DAG) snapshots with packfiles designed to compress line-by-line textual code diffs. Image binaries (JPEG/PNG) are already heavily compressed; modifying or adding binary files forces Git to store entire uncompressible blobs on every change, exploding repository size.
   * **Clone and Bandwidth Inefficiency:** Every developer or CI server running `git clone` downloads the entire repository history. Storing datasets in Git causes clones to become hundreds of megabytes or gigabytes, resulting in slow operations and frequent checkout timeouts.
   * **Platform Constraints:** Major Git hosts enforce strict hard limits (e.g., GitHub rejects individual files $>100$ MB and degrades above $2$ GB repositories).
   * **MLOps Decoupling:** DVC solves this by decoupling metadata from storage: Git versions the tiny text pointer (`.dvc`), while the large binaries reside in scalable object storage (DagsHub/S3/GCS), pullable on-demand via `dvc pull`.
