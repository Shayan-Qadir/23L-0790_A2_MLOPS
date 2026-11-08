import pytest
import os
import tensorflow as tf
from src.models.pipeline import build_dataset_from_csv, parse_image_and_label
from src.models.baseline import build_baseline_cnn

def test_data_splits_exist():
    assert os.path.exists("data/processed/train_split.csv")
    assert os.path.exists("data/processed/val_split.csv")
    assert os.path.exists("data/processed/test_split.csv")

def test_input_pipeline_shapes():
    ds = build_dataset_from_csv("data/processed/val_split.csv", batch_size=4, is_training=False)
    for images, labels in ds.take(1):
        assert images.shape == (4, 224, 224, 3), f"Expected (4, 224, 224, 3), got {images.shape}"
        assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"
        assert tf.reduce_min(images) >= 0.0
        assert tf.reduce_max(images) <= 255.0
        break

def test_baseline_cnn_forward_pass():
    model = build_baseline_cnn(input_shape=(224, 224, 3))
    dummy_input = tf.random.uniform((2, 224, 224, 3), 0, 255)
    preds = model(dummy_input, training=False)
    assert preds.shape == (2, 1), f"Expected (2, 1), got {preds.shape}"
    assert 0.0 <= preds.numpy()[0, 0] <= 1.0

