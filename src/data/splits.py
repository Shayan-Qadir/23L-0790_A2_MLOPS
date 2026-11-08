import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(audit_csv="reports/tables/dataset_audit.csv", output_dir="data/processed", seed=42):
    """
    Creates deterministic Train (70%), Val (15%), Test (15%) stratified splits.
    Excludes or deduplicates duplicate files to prevent train-test contamination.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(audit_csv)
    print(f"[*] Total valid images available: {len(df)}")
    
    # Filter out exact duplicates to prevent data leakage across splits
    # Keep the first occurrence of each unique SHA-256 hash
    initial_count = len(df)
    df_unique = df.drop_duplicates(subset=["sha256"], keep="first").copy()
    dropped_count = initial_count - len(df_unique)
    print(f"[*] Dropped {dropped_count} exact duplicate(s) to avoid data leakage.")
    print(f"[*] Unique images for splitting: {len(df_unique)}")
    
    # Stratified Train (70%) vs Temp (30%)
    train_df, temp_df = train_test_split(
        df_unique,
        test_size=0.30,
        random_state=seed,
        stratify=df_unique["class"]
    )
    
    # Stratified Val (15%) vs Test (15%) from Temp (50-50 of 30%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["class"]
    )
    
    print("\n" + "=" * 60)
    print(" SPLIT DISTRIBUTION SUMMARY")
    print("=" * 60)
    for name, split in [("Train (70%)", train_df), ("Val (15%)", val_df), ("Test (15%)", test_df)]:
        counts = split["class"].value_counts().to_dict()
        print(f"{name:<12}: Total {len(split):<4} | Class breakdown: {counts}")
        
    train_csv = out_path / "train_split.csv"
    val_csv = out_path / "val_split.csv"
    test_csv = out_path / "test_split.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\n[+] Saved split lists:")
    print(f"  - {train_csv}")
    print(f"  - {val_csv}")
    print(f"  - {test_csv}")
    print("=" * 60)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    create_splits()

