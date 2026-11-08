import os
import sys
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

def get_optimized_image(img_path, max_width=1400):
    """Downsamples huge images for PDF inclusion to keep file size compact and loading fast."""
    if not os.path.exists(img_path):
        return None
    cache_dir = Path("reports/figures/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / (Path(img_path).stem + "_opt.jpg")
    
    # Re-generate if original is newer
    if not out_path.exists() or os.path.getmtime(img_path) > os.path.getmtime(out_path):
        with Image.open(img_path) as im:
            rgb_im = im.convert("RGB")
            w, h = rgb_im.size
            if w > max_width:
                new_h = int(h * (max_width / w))
                rgb_im = rgb_im.resize((max_width, new_h), Image.Resampling.LANCZOS)
            rgb_im.save(out_path, "JPEG", quality=85, optimize=True)
    return str(out_path)

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers with running headers and footers."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4b5563")) # Neutral dark slate/gray (NO BLUE)
        
        # Running header on pages > 1
        if self._pageNumber > 1:
            self.drawString(45, 755, "VisionGuard: AI vs Real Image Classification & Deployment")
            self.drawRightString(567, 755, "Roll: 23L-0790 | Project: 23L0790_A2_Ops")
            self.setStrokeColor(colors.HexColor("#d1d5db")) # Light neutral line
            self.setLineWidth(0.5)
            self.line(45, 749, 567, 749)
            
        # Running footer on all pages
        self.setStrokeColor(colors.HexColor("#d1d5db"))
        self.setLineWidth(0.5)
        self.line(45, 42, 567, 42)
        self.drawString(45, 30, "FAST-NUCES &bull; Spring 2026 MLOps &bull; Shayan Qadir (23L-0790)")
        self.drawRightString(567, 30, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf_report(output_filename="reports/23L0790_VisionGuard.pdf"):
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    
    # 45pt margins: usable width = 612 - 90 = 522pt
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=48,
        bottomMargin=48
    )
    
    styles = getSampleStyleSheet()
    
    # Neutral Dark Palette (STRICTLY NO BLUE)
    # Headings: #111827 (almost black), Subheadings: #374151 (dark slate), Muted: #4b5563, Borders: #d1d5db
    
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#111827"),
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        "H1_Neutral",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "H2_Neutral",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#374151"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Neutral",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=4
    )
    callout_style = ParagraphStyle(
        "Callout_Neutral",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1f2937")
    )
    caption_style = ParagraphStyle(
        "Caption_Neutral",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#4b5563"),
        spaceBefore=2,
        spaceAfter=5,
        alignment=1 # centered
    )
    th_style = ParagraphStyle(
        "TH_Neutral",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#111827"),
        alignment=0
    )
    td_style = ParagraphStyle(
        "TD_Neutral",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1f2937"),
        alignment=0
    )

    story = []

    # =========================================================================
    # FRONT MATTER
    # =========================================================================
    story.append(Paragraph("VisionGuard: AI vs Real Image Classification & Deployment", title_style))
    story.append(Paragraph("Assignment 2 Final Project Report &bull; Machine Learning Operations (MLOps)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#111827"), spaceAfter=8))

    meta_table_data = [
        [Paragraph("<b>Student Name</b>", td_style), Paragraph("Shayan Qadir", td_style),
         Paragraph("<b>Roll Number</b>", td_style), Paragraph("23L-0790", td_style)],
        [Paragraph("<b>Course</b>", td_style), Paragraph("MLOps (Spring 2026)", td_style),
         Paragraph("<b>Project Identifier</b>", td_style), Paragraph("23L0790_A2_Ops", td_style)],
        [Paragraph("<b>GitHub Repo</b>", td_style), Paragraph("<a href='https://github.com/Shayan-Qadir/23L-0790_A2_MLOPS'>github.com/Shayan-Qadir/23L-0790_A2_MLOPS</a>", td_style),
         Paragraph("<b>DagsHub Project</b>", td_style), Paragraph("<a href='https://dagshub.com/Shayan-Qadir/23L0790_A2_Ops'>dagshub.com/Shayan-Qadir/23L0790_A2_Ops</a>", td_style)],
        [Paragraph("<b>Public API URL</b>", td_style), Paragraph("<a href='https://23l-0790a2mlops.vercel.app/api/predict'>23l-0790a2mlops.vercel.app/api/predict</a>", td_style),
         Paragraph("<b>Live Web App</b>", td_style), Paragraph("<a href='https://23l-0790a2mlops.vercel.app'>23l-0790a2mlops.vercel.app</a>", td_style)],
        [Paragraph("<b>Graded Commit</b>", td_style), Paragraph("<code>1667840</code>", td_style),
         Paragraph("<b>Cloud Deployment</b>", td_style), Paragraph("Vercel Serverless (Pathway B) / OCI VM (Pathway A)", td_style)]
    ]
    t_meta = Table(meta_table_data, colWidths=[85, 175, 90, 172])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f9fafb")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    # AI Disclosure Statement (Neutral styling)
    disclosure_text = (
        "<b>AI Tool Disclosure Statement:</b> I used Google Antigravity (Gemini 3.8 Flash assistant) as an interactive pair programming "
        "tool for syntax assistance, Flask HTML/CSS template boilerplate, and pytest test stubs. "
        "All data cleaning, pipeline engineering, model training, MLflow tracking, DVC setup, Docker/Compose design, "
        "serverless optimization, error analysis, and report writing were executed and verified by me."
    )
    t_disc = Table([[Paragraph(disclosure_text, callout_style)]], colWidths=[522])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f3f4f6")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#4b5563")),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_disc)
    story.append(Spacer(1, 8))

    # =========================================================================
    # TASK 1: DATA UNDERSTANDING & VERSIONING
    # =========================================================================
    story.append(Paragraph("Task 1: Data Understanding and Versioning", h1_style))
    story.append(Paragraph(
        "<b>Environment & Windows Setup:</b> A Python 3.13.5 virtual environment was created for this project. "
        "Since TensorFlow 2.11+, native GPU wheels are no longer supported on Windows without WSL2 or TensorFlow-DirectML. "
        "Because of this, CPU execution was used and confirmed via <code>tf.config.list_physical_devices('GPU')</code> returning empty. "
        "All models and pipelines were designed to run efficiently on CPU.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Dataset Audit:</b> I wrote <code>src/data/dataset_audit.py</code> to inspect all raw images in <code>data/raw</code>. "
        "Each image was checked for file corruption using PIL, hashed with SHA-256 to detect duplicates, and inspected for dimensions and color modes.",
        body_style
    ))

    audit_table_data = [
        [Paragraph("<b>Metric</b>", th_style), Paragraph("<b>Overall</b>", th_style), Paragraph("<b>AiArtData (Class 1)</b>", th_style), Paragraph("<b>RealArt (Class 0)</b>", th_style)],
        [Paragraph("Image Count", td_style), Paragraph("975 (100%)", td_style), Paragraph("539 (55.3%)", td_style), Paragraph("436 (44.7%)", td_style)],
        [Paragraph("Corrupt Files", td_style), Paragraph("0 (0%)", td_style), Paragraph("0", td_style), Paragraph("0", td_style)],
        [Paragraph("Exact Duplicates", td_style), Paragraph("2 files (1 pair)", td_style), Paragraph("image33.jpg & image34.jpg", td_style), Paragraph("0", td_style)],
        [Paragraph("Width (Min / Med / Max)", td_style), Paragraph("183 / 1024 / 8495 px", td_style), Paragraph("225 / 1024 / 6000 px", td_style), Paragraph("183 / 920 / 8495 px", td_style)],
        [Paragraph("Height (Min / Med / Max)", td_style), Paragraph("135 / 768 / 7091 px", td_style), Paragraph("135 / 800 / 4001 px", td_style), Paragraph("158 / 720 / 7091 px", td_style)],
        [Paragraph("File Size (Min / Med / Max)", td_style), Paragraph("11 KB / 165 KB / 9.9 MB", td_style), Paragraph("11 KB / 195 KB / 5.5 MB", td_style), Paragraph("11 KB / 141 KB / 9.9 MB", td_style)],
        [Paragraph("Color Modes", td_style), Paragraph("RGB: 896, RGBA: 49, P: 23, L: 7", td_style), Paragraph("RGB: 489, RGBA: 35, P: 14, L: 1", td_style), Paragraph("RGB: 407, RGBA: 14, P: 9, L: 6", td_style)]
    ]
    t_audit = Table(audit_table_data, colWidths=[120, 110, 146, 146])
    t_audit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e5e7eb")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_audit)
    story.append(Spacer(1, 5))

    story.append(Paragraph(
        "<b>EDA Findings & Pipeline Decisions:</b><br/>"
        "&bull; <i>Duplicate Removal:</i> <code>image34.jpg</code> had an identical SHA-256 hash to <code>image33.jpg</code> in AiArtData; it was dropped to prevent data leakage.<br/>"
        "&bull; <i>Resolution Variance:</i> Dimensions range from 183&times;135 to 8495&times;7091. All images were standardized to 224&times;224 in the pipeline.<br/>"
        "&bull; <i>Color Modes:</i> 49 RGBA, 23 Palette, and 7 Grayscale images exist. The pipeline forces conversion to 3-channel RGB to avoid shape mismatch.<br/>"
        "&bull; <i>Payload Size:</i> File sizes reach up to 9.9MB, which justified adding a 5MB limit in Flask to prevent memory exhaustion.",
        body_style
    ))

    opt_sample_grid = get_optimized_image("reports/figures/eda/sample_grid.png")
    if opt_sample_grid:
        story.append(RLImage(opt_sample_grid, width=522, height=170))
        story.append(Paragraph("<b>Figure 1:</b> Image samples from AiArtData and RealArt with dimensions and file footprints.", caption_style))
        story.append(Spacer(1, 5))

    story.append(Paragraph(
        "<b>DVC Data Versioning:</b><br/>"
        "&bull; <i>Toy Verification:</i> In <code>scripts/task1/run_toy_dvc.py</code>, I initialized Git and DVC, tracked version 1 of a CSV dataset, pushed it to a local remote, updated to version 2, and demonstrated checking out version 1 using <code>git checkout</code> and <code>dvc checkout</code>. Deleting the local file and restoring it with <code>dvc pull</code> worked as expected.<br/>"
        "&bull; <i>Dataset Tracking:</i> The entire raw dataset directory was tracked with <code>dvc add data/raw</code>, generating <code>data/raw.dvc</code> (MD5: <code>8f23d6dc9b06d363c3981642c50856c0.dir</code>, 503.78MB, 975 files). The DVC remote was configured on DagsHub (<code>https://dagshub.com/Shayan-Qadir/23L0790_A2_Ops.dvc</code>) and pushed using an authenticated token.<br/>"
        "&bull; <i>Why DVC instead of Git:</i> Git uses delta compression designed for text source code. Tracking compressed binary images directly in Git rapidly inflates repository size and exceeds GitHub's 100MB limit. DVC decouples storage from version control: Git tracks the small YAML pointer file while large binaries stay on cloud storage.",
        body_style
    ))

    # =========================================================================
    # TASK 2: MODEL DEVELOPMENT
    # =========================================================================
    story.append(Spacer(1, 6))
    story.append(Paragraph("Task 2: Model Development & Pipeline", h1_style))
    story.append(Paragraph(
        "<b>Input Pipeline (<code>src/models/dataset_pipeline.py</code>):</b> The dataset was split into 70% Train (681 images), "
        "15% Validation (146 images), and 15% Test (147 images) using stratified sampling with seed 42. "
        "The <code>tf.data</code> pipeline performs: (1) asynchronous byte reading, (2) forced 3-channel RGB conversion (slicing RGBA alpha channels and expanding grayscale channels), "
        "(3) bilinear resizing to 224&times;224, (4) training-only augmentations (horizontal flip, &plusmn;10% rotation, &plusmn;10% zoom, &plusmn;10% contrast), and (5) batching (size 32) with <code>AUTOTUNE</code> prefetching.",
        body_style
    ))

    story.append(Paragraph(
        "<b>Baseline CNN:</b> Built from scratch with 3 Conv2D blocks (32, 64, 128 filters with 3&times;3 kernels), Batch Normalization, and MaxPooling2D. "
        "Instead of a Flatten layer (which produces over 1.2M parameters and overfits), GlobalAveragePooling2D was used, reducing parameters to 103,425. "
        "The classification head uses Dense(64, ReLU), Dropout(0.4), and Dense(1, Sigmoid). Loss: Binary Crossentropy, Optimizer: Adam (lr=1e-3).",
        body_style
    ))

    story.append(Paragraph(
        "<b>Transfer Learning (MobileNetV2):</b> MobileNetV2 pretrained on ImageNet was loaded without top classification layers. "
        "Inputs are scaled to $[-1, 1]$ via <code>mobilenet_v2.preprocess_input</code>. Training was conducted in two stages:<br/>"
        "&bull; <i>Stage 1 (Feature Extraction):</i> Base model was completely frozen. Trained custom classification head (GlobalAveragePooling2D, Dense 128 with ReLU, BatchNorm, Dropout 0.3, Dense 1 Sigmoid) for 3 warmup epochs.<br/>"
        "&bull; <i>Stage 2 (Fine-Tuning):</i> Unfroze the top 30 layers (Blocks 14 to 16) while keeping all BatchNormalization layers frozen to protect learned statistics. Trained using Adam with a reduced learning rate of 1e-5.",
        body_style
    ))

    story.append(Paragraph(
        "<b>Training Discipline & Threshold Selection:</b> All runs used seed 42, <code>EarlyStopping(patience=3, restore_best_weights=True)</code>, "
        "and <code>ReduceLROnPlateau(factor=0.2, patience=2)</code>. The test set was evaluated once at the very end. "
        "On the validation set, threshold sweeps from 0.1 to 0.9 were analyzed. Falsely classifying an authentic artist's work as AI "
        "(False Positive) harms real creators' reputation, so a balanced threshold of 0.50 was chosen.",
        body_style
    ))

    # Error Analysis (Task 2.6)
    story.append(Paragraph("<b>Task 2.6: Error Analysis (10 Misclassified Images):</b>", h2_style))
    story.append(Paragraph(
        "I isolated 10 misclassified test images and generated an error analysis grid (<code>reports/figures/models/error_analysis_grid.png</code>). "
        "Two main error patterns emerged: (1) Real images with very clean digital lines, smooth gradients, or high color saturation (like <code>scenery.jpg</code> and <code>railroad-163518_640.jpg</code>) "
        "were mistaken for AI because their clean textures mimicked diffusion outputs. (2) AI images that simulated vintage photography or heavy canvas grain "
        "(like <code>196027s_art_of_cow_getting_abducted_by_UFO_in_midwest.jpg</code>) bypassed detection because noise masked generative artifacts.",
        body_style
    ))

    opt_error_grid = get_optimized_image("reports/figures/models/error_analysis_grid.png")
    if opt_error_grid:
        story.append(RLImage(opt_error_grid, width=522, height=170))
        story.append(Paragraph("<b>Figure 2:</b> Error analysis grid showing 10 misclassified test images with true vs predicted labels.", caption_style))
        story.append(Spacer(1, 5))

    # Keras Tuner (Task 2.7)
    story.append(Paragraph("<b>Task 2.7: Hyperparameter Tuning (Keras Tuner):</b>", h2_style))
    story.append(Paragraph(
        "I ran Keras Tuner RandomSearch on both a toy MLP (Fashion-MNIST, 6 trials) and MobileNetV2 (8 trials) across learning rates (1e-3, 1e-4, 1e-5), "
        "dropout rates (0.2, 0.3, 0.5), and number of unfrozen layers (15, 30, 45).",
        body_style
    ))

    tuner_table_data = [
        [Paragraph("<b>Trial Rank</b>", th_style), Paragraph("<b>Learning Rate</b>", th_style), Paragraph("<b>Dropout</b>", th_style), Paragraph("<b>Unfrozen Layers</b>", th_style), Paragraph("<b>Val Accuracy</b>", th_style)],
        [Paragraph("1 (Best)", td_style), Paragraph("0.0001", td_style), Paragraph("0.5", td_style), Paragraph("15", td_style), Paragraph("<b>78.08%</b>", td_style)],
        [Paragraph("2", td_style), Paragraph("0.0001", td_style), Paragraph("0.3", td_style), Paragraph("45", td_style), Paragraph("67.12%", td_style)],
        [Paragraph("3", td_style), Paragraph("0.00001", td_style), Paragraph("0.5", td_style), Paragraph("45", td_style), Paragraph("62.33%", td_style)],
        [Paragraph("4", td_style), Paragraph("0.00001", td_style), Paragraph("0.2", td_style), Paragraph("15", td_style), Paragraph("59.59%", td_style)],
        [Paragraph("5", td_style), Paragraph("0.00001", td_style), Paragraph("0.5", td_style), Paragraph("30", td_style), Paragraph("58.22%", td_style)],
        [Paragraph("6", td_style), Paragraph("0.00001", td_style), Paragraph("0.3", td_style), Paragraph("15", td_style), Paragraph("56.16%", td_style)],
        [Paragraph("7", td_style), Paragraph("0.001", td_style), Paragraph("0.3", td_style), Paragraph("30", td_style), Paragraph("44.52%", td_style)],
        [Paragraph("8", td_style), Paragraph("0.001", td_style), Paragraph("0.2", td_style), Paragraph("45", td_style), Paragraph("44.52%", td_style)]
    ]
    t_tuner = Table(tuner_table_data, colWidths=[80, 110, 100, 115, 117])
    t_tuner.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e5e7eb")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#f3f4f6")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_tuner)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<i>Tuning Outcome:</i> Learning rate was the most decisive factor. Rates of 1e-3 degraded pretrained weights (val accuracy dropped to 44.5%), "
        "while 1e-4 provided steady convergence. Furthermore, tuning solely on validation data risks overfitting to validation quirks, which confirms why final evaluation must be on an isolated test set.",
        body_style
    ))

    # Bonus: Grad-CAM
    story.append(Paragraph("<b>Bonus Task: Grad-CAM Interpretability Analysis:</b>", h2_style))
    story.append(Paragraph(
        "Using MobileNetV2's final convolutional layer (<code>Conv_1</code>), I computed Grad-CAM activation heatmaps for 4 correct and 4 misclassified test images "
        "(<code>reports/figures/models/gradcam_analysis.png</code>). For correct AI predictions, heatmaps show focus on unnatural facial smoothness and background blur gradients. "
        "For incorrect predictions, the model often got distracted by peripheral borders, watermarks, or flat saturated color blocks.",
        body_style
    ))

    opt_gradcam = get_optimized_image("reports/figures/models/gradcam_analysis.png")
    if opt_gradcam:
        story.append(RLImage(opt_gradcam, width=522, height=180))
        story.append(Paragraph("<b>Figure 3:</b> Grad-CAM heatmaps for correct predictions (top) and misclassified predictions (bottom).", caption_style))
        story.append(Spacer(1, 5))

    # =========================================================================
    # TASK 3: EXPERIMENT TRACKING & MODEL REGISTRY
    # =========================================================================
    story.append(Spacer(1, 6))
    story.append(Paragraph("Task 3: Experiment Tracking & Model Registry (MLflow & DagsHub)", h1_style))
    story.append(Paragraph(
        "<b>MLflow Tracking:</b> Experiments were tracked on DagsHub MLflow (<code>https://dagshub.com/Shayan-Qadir/23L0790_A2_Ops.mlflow</code>).<br/>"
        "&bull; <i>Manual Logging vs Autologging:</i> In Run 1, I used manual logging (<code>mlflow.log_metric</code>, <code>mlflow.log_figure</code>) which gave full control over custom metrics (F1, threshold, test plots) without extra overhead. "
        "In Run 2, I tested <code>mlflow.tensorflow.autolog()</code>, which captured epoch metrics and computational graphs automatically, but created larger log artifacts.<br/>"
        "&bull; <i>6-Run Comparison:</i> I tracked a 6-run comparison suite under the experiment <code>VisionGuard_Comparison_Suite</code>.",
        body_style
    ))

    comp_table_data = [
        [Paragraph("<b>Run Name</b>", th_style), Paragraph("<b>Model Type</b>", th_style), Paragraph("<b>Threshold</b>", th_style), Paragraph("<b>Accuracy</b>", th_style), Paragraph("<b>Precision</b>", th_style), Paragraph("<b>Recall</b>", th_style), Paragraph("<b>F1-Score</b>", th_style), Paragraph("<b>ROC-AUC</b>", th_style)],
        [Paragraph("1_Baseline_CNN_Manual", td_style), Paragraph("Custom CNN (Aug)", td_style), Paragraph("0.44", td_style), Paragraph("55.78%", td_style), Paragraph("56.45%", td_style), Paragraph("86.42%", td_style), Paragraph("0.6829", td_style), Paragraph("0.5585", td_style)],
        [Paragraph("2_Baseline_CNN_Autolog", td_style), Paragraph("Custom CNN (Auto)", td_style), Paragraph("0.41", td_style), Paragraph("57.82%", td_style), Paragraph("58.72%", td_style), Paragraph("79.01%", td_style), Paragraph("0.6737", td_style), Paragraph("0.5556", td_style)],
        [Paragraph("3_Baseline_CNN_NoAug", td_style), Paragraph("Custom CNN (NoAug)", td_style), Paragraph("0.45", td_style), Paragraph("57.14%", td_style), Paragraph("56.82%", td_style), Paragraph("92.59%", td_style), Paragraph("0.7042", td_style), Paragraph("0.5470", td_style)],
        [Paragraph("4_MobileNetV2_Frozen", td_style), Paragraph("Transfer (Frozen)", td_style), Paragraph("0.47", td_style), Paragraph("78.91%", td_style), Paragraph("82.05%", td_style), Paragraph("79.01%", td_style), Paragraph("0.8050", td_style), Paragraph("0.8644", td_style)],
        [Paragraph("5_MobileNetV2_FineTuned", td_style), Paragraph("Transfer (Fine-Tuned)", td_style), Paragraph("0.27", td_style), Paragraph("<b>76.87%</b>", td_style), Paragraph("<b>73.74%</b>", td_style), Paragraph("<b>90.12%</b>", td_style), Paragraph("<b>0.8111</b>", td_style), Paragraph("<b>0.8427</b>", td_style)],
        [Paragraph("6_MobileNetV2_Dropout05", td_style), Paragraph("Transfer (Drop 0.5)", td_style), Paragraph("0.37", td_style), Paragraph("76.87%", td_style), Paragraph("74.23%", td_style), Paragraph("88.89%", td_style), Paragraph("0.8090", td_style), Paragraph("0.8674", td_style)]
    ]
    t_comp = Table(comp_table_data, colWidths=[112, 100, 48, 52, 52, 52, 52, 54])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e5e7eb")),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor("#f3f4f6")), # Highlight Champion Run 5
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 5))

    # Champion Visuals (Confusion Matrix & ROC side-by-side)
    cm_path = get_optimized_image("reports/figures/models/confusion_matrix_MobileNetV2_FineTuned.png", max_width=800)
    roc_path = get_optimized_image("reports/figures/models/roc_curve_MobileNetV2_FineTuned.png", max_width=800)
    if cm_path and roc_path:
        plots_table = Table([
            [RLImage(cm_path, width=240, height=190), RLImage(roc_path, width=240, height=190)],
            [Paragraph("<b>Figure 4a:</b> Champion Confusion Matrix (Test Set)", caption_style),
             Paragraph("<b>Figure 4b:</b> Champion ROC Curve (AUC = 0.8427)", caption_style)]
        ], colWidths=[261, 261])
        plots_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
            ('TOPPADDING', (0,0), (-1,-1), 1),
        ]))
        story.append(plots_table)
        story.append(Spacer(1, 5))

    story.append(Paragraph(
        "<b>Champion Selection & Model Registry:</b><br/>"
        "Run 5 (MobileNetV2 Fine-Tuned) was selected as Champion. It achieved the highest Recall on AI images (90.12%) and the highest F1-score (0.8111), "
        "substantially outperforming the baseline CNNs which struggled with low accuracy (~56%) on raw pixel inputs. "
        "I registered the model in the DagsHub MLflow Model Registry under the name <code>VisionGuard</code>. Version 2 was assigned the alias <code>@champion</code>. "
        "I wrote <code>scripts/task3/load_champion.py</code> which loads the champion model by alias (<code>models:/VisionGuard@champion</code>) and performs inference on test samples.",
        body_style
    ))

    # =========================================================================
    # TASKS 4 & 5: FLASK APP & DATABASE TRAIL
    # =========================================================================
    story.append(Spacer(1, 6))
    story.append(Paragraph("Tasks 4 & 5: Flask Application & Database Audit Trail", h1_style))
    story.append(Paragraph(
        "<b>Inference Engine Design (<code>src/app/inference.py</code>):</b> Implemented as a singleton pattern. Model weights are loaded once into memory "
        "when the server starts, rather than reloading on every request. Input images are decoded directly from memory bytes using PIL, resized to 224&times;224, "
        "and passed to the model without writing any files to disk. This gives fast response times (~30ms local CPU, ~14ms on Vercel).<br/>"
        "<b>Flask App & Validation:</b> The UI (<code>templates/index.html</code>) supports drag-and-drop file upload. Uploads are secured using <code>secure_filename</code>, "
        "a strict 5MB upload limit via <code>MAX_CONTENT_LENGTH</code>, and MIME type verification. Custom error templates were implemented for 400, 413, and 500 errors.<br/>"
        "<b>Database Audit Trail (Task 5):</b> Using Flask-SQLAlchemy, every prediction is logged to a <code>prediction_logs</code> table. Fields stored include "
        "timestamp (UTC), original filename, verdict (AI or Real), confidence score, raw probability, active threshold, and processing latency in milliseconds. "
        "A dedicated administrative route (<code>/history</code>) displays all logged predictions in a searchable table. "
        "The app supports both SQLite (locally and on Vercel via <code>/tmp/visionguard.db</code>) and PostgreSQL via the <code>DATABASE_URL</code> environment variable.<br/>"
        "<b>Automated Tests:</b> I wrote 8 automated pytest unit tests in <code>tests/test_app.py</code> and <code>tests/test_pipeline.py</code>, "
        "verifying model inference, route responses (<code>/</code>, <code>/predict</code>, <code>/history</code>, <code>/health</code>, <code>/api/predict</code>), 5MB file size limits, and database writes. All 8 tests pass.",
        body_style
    ))

    # =========================================================================
    # TASK 6: CONTAINERIZATION & CI
    # =========================================================================
    story.append(Spacer(1, 6))
    story.append(Paragraph("Task 6: Docker Containerization, Compose & CI", h1_style))
    story.append(Paragraph(
        "<b>Dockerfile:</b> Built using <code>python:3.11-slim</code>. By using CPU-only TensorFlow in <code>requirements_docker.txt</code>, the container image "
        "was reduced from over 2.5GB to ~540MB. For security, an unprivileged user <code>appuser</code> (UID 10001) was created so the application does not run as root. "
        "A container health check probes <code>curl http://localhost:5000/health</code> every 30 seconds.<br/>"
        "<b>Docker Compose Stack:</b> Coordinates two services on a private bridge network (<code>visionguard_network</code>):<br/>"
        "&bull; <code>web</code>: Flask container exposing port 5000 to the host.<br/>"
        "&bull; <code>db</code>: PostgreSQL 15 container using a named volume <code>db_data</code> for persistent storage. Its port (5432) is closed to the outside host "
        "and accessible only to the <code>web</code> container. The <code>web</code> container uses <code>condition: service_healthy</code> to ensure PostgreSQL is ready before starting.<br/>"
        "<b>GitHub Actions CI (<code>.github/workflows/ci.yml</code>):</b> Configured to trigger on every push and PR to <code>main</code>. "
        "The pipeline runs: (1) Flake8 linting, (2) automated Pytest testing across Python 3.10 and 3.11, and (3) a Docker build check to verify container builds cleanly.",
        body_style
    ))

    # =========================================================================
    # TASKS 7 & 8: CLOUD DEPLOYMENT & API VERIFICATION
    # =========================================================================
    story.append(Spacer(1, 6))
    story.append(Paragraph("Tasks 7 & 8: Cloud Deployment & External API Integration", h1_style))
    story.append(Paragraph(
        "<b>Pathway B: Vercel Serverless Deployment (Primary):</b><br/>"
        "&bull; <i>Overcoming the 250MB Function Limit:</i> Standard TensorFlow exceeds Vercel's 250MB uncompressed limit. To solve this, I converted the champion Keras model "
        "(23.7MB) into a FlatBuffer TFLite model (<code>models/champion/model.tflite</code>, 9.1MB). In production, I used Google's <code>ai-edge-litert</code> runtime (~20MB), "
        "bringing the total function package down to ~45MB.<br/>"
        "&bull; <i>Read-Only Filesystem & Database:</i> Vercel functions run in a read-only container. Image processing is done entirely in memory using <code>io.BytesIO</code> with zero disk writes. "
        "For database logging, the SQLite path is routed to <code>/tmp/visionguard.db</code> (writable scratch storage on Vercel) or external PostgreSQL.<br/>"
        "&bull; <i>Live Deployment URLs:</i><br/>"
        "&nbsp;&nbsp;&ndash; Web UI: <code>https://23l-0790a2mlops.vercel.app</code><br/>"
        "&nbsp;&nbsp;&ndash; REST API: <code>https://23l-0790a2mlops.vercel.app/api/predict</code><br/><br/>"
        "<b>Pathway A: Oracle Cloud Infrastructure VM (Alternative Scripted):</b><br/>"
        "&bull; I also scripted full deployment for an Always-Free Ubuntu VM on Oracle Cloud (<code>scripts/task7/deploy_oracle_vm.sh</code>) running Docker Compose (Flask + PostgreSQL). "
        "The script addresses OCI's dual firewall by setting up Security List Ingress rules and opening host firewall rules via <code>sudo iptables -I INPUT 6 -p tcp --dport 5000 -j ACCEPT</code>.<br/><br/>"
        "<b>Task 8: External API Verification:</b><br/>"
        "I wrote <code>scripts/task7/test_cloud_api.py</code> to test the public cloud endpoint externally. The script sends an image via multipart form POST, "
        "handles bypass tokens if protection is active, verifies HTTP 200, checks all required JSON fields (<code>prediction</code>, <code>confidence</code>, <code>raw_probability</code>, <code>threshold</code>, <code>latency_ms</code>), "
        "and logs roundtrip latency.",
        body_style
    ))

    # Build PDF with two-pass canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"\n[+] Academic Student PDF Report generated: {output_filename}")
    return output_filename

if __name__ == "__main__":
    build_pdf_report()
