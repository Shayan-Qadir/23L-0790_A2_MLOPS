import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

CLASS_TO_LABEL = {"RealArt": 0.0, "AiArtData": 1.0}
LABEL_TO_CLASS = {0.0: "RealArt (Real)", 1.0: "AiArtData (AI-Generated)"}

def parse_image_and_label(filepath, label, img_size=(224, 224)):
    """
    Decodes image from path and resizes to target shape.
    Enforces 3-channel RGB representation across all formats.
    """
    # 1. Read raw binary bytes
    img_bytes = tf.io.read_file(filepath)
    
    # 2. Decode image (channels=3 converts grayscale L, palette P, and RGBA to standard 3-channel RGB)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    
    # 3. Explicit shape assertion for graph tracing
    img.set_shape([None, None, 3])
    
    # 4. Resize to uniform spatial dimensions (224x224)
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    
    return img, label

def get_augmentation_layer():
    """Builds Keras random data augmentation pipeline for training."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")

def build_dataset_from_csv(csv_path, batch_size=32, img_size=(224, 224), is_training=False, shuffle_buffer=1000):
    """
    Constructs a high-performance tf.data.Dataset pipeline.
    Stages: Read Split -> Tensor Slices -> Decode & Resize -> (Augment) -> Batch -> Prefetch
    """
    df = pd.read_csv(csv_path)
    filepaths = df["filepath"].values
    labels = df["class"].map(CLASS_TO_LABEL).values.astype(np.float32)
    
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(df), shuffle_buffer), seed=42, reshuffle_each_iteration=True)
        
    # Map decoding and resizing with parallel calls
    ds = ds.map(
        lambda path, lbl: parse_image_and_label(path, lbl, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch before augmentation
    ds = ds.batch(batch_size, drop_remainder=False)
    
    # Apply data augmentation only on training batches
    if is_training:
        augmentation = get_augmentation_layer()
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    # Prefetch in GPU/CPU background buffer
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def save_augmented_batch_figure(train_csv="data/processed/train_split.csv", output_path="reports/figures/models/augmented_batch.png"):
    """Extracts and saves a grid of images from one augmented training batch."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    train_ds = build_dataset_from_csv(train_csv, batch_size=16, is_training=True)
    
    for images, labels in train_ds.take(1):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle("Task 2.1: Sample Augmented Training Batch (tf.data Pipeline)", fontsize=14, weight="bold")
        
        for idx in range(16):
            r, c = idx // 4, idx % 4
            ax = axes[r, c]
            # Convert float32 [0, 255] back to uint8 for display
            img_np = np.clip(images[idx].numpy(), 0, 255).astype(np.uint8)
            label_val = float(labels[idx].numpy())
            class_name = LABEL_TO_CLASS.get(label_val, "Unknown")
            
            ax.imshow(img_np)
            ax.set_title(f"{class_name}\nShape: {img_np.shape}", fontsize=9)
            ax.axis("off")
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Augmented batch figure saved to: {output_path}")
        break

if __name__ == "__main__":
    save_augmented_batch_figure()

