# Task 3: Experiment Tracking with MLflow and DagsHub

## 3.1 & 3.2 MLflow Fundamentals & DagsHub Integration

### Core MLOps Definitions
* **Experiment:** A logical organizational container that groups multiple associated training executions focused on solving a specific machine learning objective. It allows engineers to compare varying model families, architectures, and iterations under a unified namespace.
* **Run:** A distinct, individual execution instance of a training script within an experiment. A run records the complete provenance of that specific trial—including code commit, hardware environment, inputs, hyperparameters, trajectory curves, and outputs.
* **Parameter:** An immutable input configuration value chosen prior to model execution that directs the behavior of the training pipeline or architecture (e.g., learning rate, optimizer choice, batch size, dropout probability).
* **Metric:** A scalar numerical measurement recorded during training (at epoch/step intervals) or during post-training evaluation to assess quantitative model performance (e.g., training loss, validation accuracy, precision, recall, ROC-AUC).
* **Artifact:** Any persistent unstructured file or binary asset generated as an output of a run and preserved in artifact storage for auditing, inspection, or downstream serving (e.g., trained weights `.keras`, serialized pipelines, diagnostic plots, split manifests, environment dependencies).
* **Tag:** Descriptive metadata attached to a run as key-value string pairs, designed for filtering, categorization, and operational traceability (e.g., `git_commit`, `dvc_version`, `developer`, `model_type`, `framework`).

---

### Verification: Local & DagsHub Tracking
1. **Local Tracking Backend:**
   * Backend Store: Modern relational SQLite storage (`sqlite:///mlflow.db`).
   * Compared two training runs with distinct learning rates:
     * Run 1 (`LR = 0.01`): Final Validation Loss: 0.4830 | Final Validation Accuracy: 82.60%
     * Run 2 (`LR = 0.001`): Final Validation Loss: 0.5446 | Final Validation Accuracy: 80.60%
2. **DagsHub Remote Tracking:**
   * Remote Tracking URI: `https://dagshub.com/Shayan-Qadir/23L0790_A2_Ops.mlflow`
   * Credentials dynamically loaded from `.env` via `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`.
   * Experiment Name: `Toy_MLflow_Lab_DagsHub`
   * **Active Run Links on DagsHub:**
     * Run 1 (`LR = 0.01`): [Run a8f2eb16](https://dagshub.com/Shayan-Qadir/23L0790_A2_Ops.mlflow/#/experiments/0/runs/a8f2eb1601204492a683715b3572f8e8)
     * Run 2 (`LR = 0.001`): [Run bbd2ac35](https://dagshub.com/Shayan-Qadir/23L0790_A2_Ops.mlflow/#/experiments/0/runs/bbd2ac35bc07418089dde8fa345475c4)

