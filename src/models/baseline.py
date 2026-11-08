import tensorflow as tf
import pandas as pd

def build_baseline_cnn(input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Constructs a custom convolutional neural network from scratch.
    
    Architecture:
    - Input: (224, 224, 3)
    - Rescaling: Normalizes pixel intensities from [0, 255] to [0, 1]
    - Conv Block 1: Conv2D(32, 3x3) -> BatchNormalization -> MaxPooling2D(2x2)
    - Conv Block 2: Conv2D(64, 3x3) -> BatchNormalization -> MaxPooling2D(2x2)
    - Conv Block 3: Conv2D(128, 3x3) -> BatchNormalization -> MaxPooling2D(2x2)
    - Head: GlobalAveragePooling2D -> Dense(64, ReLU) -> Dropout(0.4) -> Dense(1, Sigmoid)
    
    Justification:
    - Conv2D captures hierarchical spatial features (edges -> textures -> semantic shapes).
    - BatchNormalization stabilizes internal covariate shift and accelerates convergence.
    - MaxPooling reduces spatial dimensions, providing translation invariance.
    - GlobalAveragePooling2D condenses spatial feature maps into a 1D vector while drastically
      reducing parameter count compared to Flatten(), preventing overfitting.
    - Dropout(0.4) regularizes dense weights.
    - Sigmoid output produces well-calibrated class probability [0, 1] for binary classification.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling_norm")(inputs)
    
    # Conv Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    
    # Conv Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    # Conv Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)
    
    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="VisionGuard_Baseline_CNN")
    
    # Optimizer & Loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model

def export_model_summary_table(model, output_path="reports/tables/baseline_cnn_summary.md"):
    """Extracts output shape and parameter count per layer into a markdown table."""
    records = []
    for layer in model.layers:
        records.append({
            "Layer Name": layer.name,
            "Layer Type": layer.__class__.__name__,
            "Output Shape": str(layer.output.shape),
            "Param Count": f"{layer.count_params():,}"
        })
    df = pd.DataFrame(records)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Task 2.2: Baseline CNN Layer-by-Layer Architecture\n\n")
        f.write(df.to_markdown(index=False))
        f.write(f"\n\n**Total Parameters:** {model.count_params():,}\n")
        f.write(f"**Trainable Parameters:** {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}\n")
        f.write(f"**Non-trainable Parameters:** {sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}\n")
    print(f"[+] Baseline summary table saved to: {output_path}")
    return df

if __name__ == "__main__":
    model = build_baseline_cnn()
    model.summary()
    export_model_summary_table(model)

