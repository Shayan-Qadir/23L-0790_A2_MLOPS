import os
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def audit_dataset(data_dir="data/raw", output_csv="reports/tables/dataset_audit.csv"):
    """
    Walks dataset and extracts:
    - class, file size, width, height, colour mode, format, sha256 hash.
    Detects unreadable files and exact duplicates.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    records = []
    corrupt_files = []
    
    print(f"[*] Starting dataset audit on: {data_path.resolve()}")
    
    # Iterate through class folders
    for class_dir in sorted([d for d in data_path.iterdir() if d.is_dir()]):
        class_name = class_dir.name
        image_files = [f for f in class_dir.iterdir() if f.is_file()]
        print(f" -> Scanning class '{class_name}' ({len(image_files)} files)...")
        
        for file_path in image_files:
            file_size_bytes = file_path.stat().st_size
            file_size_kb = file_size_bytes / 1024.0
            
            try:
                content_hash = compute_sha256(file_path)
            except Exception as e:
                corrupt_files.append({"filepath": str(file_path), "class": class_name, "error": f"Read error: {e}"})
                continue
                
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify integrity
                # Reopen to read metadata (verify closes image)
                with Image.open(file_path) as img:
                    width, height = img.size
                    color_mode = img.mode
                    img_format = img.format if img.format else file_path.suffix.upper().replace(".", "")
                    
                records.append({
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "class": class_name,
                    "file_size_bytes": file_size_bytes,
                    "file_size_kb": file_size_kb,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 4) if height > 0 else 0,
                    "color_mode": color_mode,
                    "format": img_format,
                    "sha256": content_hash,
                    "is_corrupt": False
                })
            except Exception as e:
                corrupt_files.append({
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "class": class_name,
                    "file_size_bytes": file_size_bytes,
                    "file_size_kb": file_size_kb,
                    "width": None,
                    "height": None,
                    "aspect_ratio": None,
                    "color_mode": None,
                    "format": None,
                    "sha256": content_hash,
                    "is_corrupt": True,
                    "error": str(e)
                })

    df = pd.DataFrame(records)
    
    # Duplicate detection by SHA-256
    hash_counts = df["sha256"].value_counts()
    duplicate_hashes = hash_counts[hash_counts > 1].index.tolist()
    df["is_duplicate"] = df["sha256"].isin(duplicate_hashes)
    
    # Save full audit table
    df.to_csv(output_csv, index=False)
    print(f"\n[+] Audit completed. Detailed records saved to: {output_csv}")
    
    # Generate summary statistics
    print("\n" + "=" * 70)
    print("                    DATASET AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total Images Scanned : {len(df) + len(corrupt_files)}")
    print(f"Valid Readable Images: {len(df)}")
    print(f"Corrupt Images Found : {len(corrupt_files)}")
    print(f"Exact Duplicate Files: {df['is_duplicate'].sum()} (unique duplicate hashes: {len(duplicate_hashes)})")
    print("-" * 70)
    
    # Class breakdown
    print("\n[COUNTS PER CLASS]")
    class_counts = df["class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    class_counts["Percentage"] = (class_counts["Count"] / len(df) * 100).round(2).astype(str) + "%"
    print(class_counts.to_string(index=False))
    
    # Dimensions & Size stats (overall & per-class)
    print("\n[DIMENSIONS & FILE SIZE SUMMARY (MIN, MEDIAN, MAX)]")
    metrics_summary = []
    
    for cls in ["Overall"] + list(df["class"].unique()):
        sub_df = df if cls == "Overall" else df[df["class"] == cls]
        metrics_summary.append({
            "Subset": cls,
            "Count": len(sub_df),
            "Width (Min/Med/Max)": f"{sub_df['width'].min()} / {sub_df['width'].median():.0f} / {sub_df['width'].max()}",
            "Height (Min/Med/Max)": f"{sub_df['height'].min()} / {sub_df['height'].median():.0f} / {sub_df['height'].max()}",
            "Size KB (Min/Med/Max)": f"{sub_df['file_size_kb'].min():.1f} / {sub_df['file_size_kb'].median():.1f} / {sub_df['file_size_kb'].max():.1f}"
        })
        
    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.to_string(index=False))
    
    # Color modes & formats
    print("\n[COLOR MODES]")
    print(df.groupby(["class", "color_mode"]).size().unstack(fill_value=0).to_string())
    
    print("\n[FILE FORMATS]")
    print(df.groupby(["class", "format"]).size().unstack(fill_value=0).to_string())
    
    # Save summary tables to markdown for the report
    md_summary_path = "reports/tables/audit_summary.md"
    with open(md_summary_path, "w", encoding="utf-8") as f:
        f.write("# Task 1.2: Dataset Audit Summary\n\n")
        f.write("### Class Counts\n\n")
        f.write(class_counts.to_markdown(index=False))
        f.write("\n\n### Metric Distributions (Min, Median, Max)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n### Integrity and Duplication\n\n")
        f.write(f"- **Total Scanned**: {len(df) + len(corrupt_files)}\n")
        f.write(f"- **Valid Images**: {len(df)}\n")
        f.write(f"- **Corrupt Files**: {len(corrupt_files)}\n")
        f.write(f"- **Exact Duplicates (SHA-256 match)**: {df['is_duplicate'].sum()}\n")
        f.write("\n### Color Modes\n\n")
        f.write(df.groupby(["class", "color_mode"]).size().unstack(fill_value=0).to_markdown())
        f.write("\n\n### Formats\n\n")
        f.write(df.groupby(["class", "format"]).size().unstack(fill_value=0).to_markdown())
        f.write("\n")
        
    print(f"\n[+] Markdown summary table written to: {md_summary_path}")
    print("=" * 70)
    
    if len(duplicate_hashes) > 0:
        print("\n[DUPLICATE DETAILS]")
        dup_df = df[df["is_duplicate"]][["filename", "class", "sha256"]].sort_values("sha256")
        print(dup_df.head(10).to_string(index=False))
        
    return df, summary_df

if __name__ == "__main__":
    audit_dataset()

