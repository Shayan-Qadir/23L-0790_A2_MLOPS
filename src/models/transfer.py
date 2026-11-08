import tensorflow as tf
import pandas as pd

def build_transfer_model(input_shape=(224, 224, 3), learning_rate=1e-3, dropout_rate=0.3):
    """
    Constructs a transfer learning model utilizing MobileNetV2 pretrained on ImageNet.
    
    Preprocessing Rationale:
    - MobileNetV2 expects pixel intensities scaled to [-1.0, 1.0].
    - We utilize tf.keras.applications.mobilenet_v2.preprocess_input to match
      the exact distribution the network was trained on during ImageNet pretraining.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # 1. Official MobileNetV2 Preprocessing Layer (scales [0, 255] -> [-1, 1])
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # 2. Pretrained Backbone (without top classifier)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # Initial Stage: Freeze all weights in the base backbone
    base_model.trainable = False
    
    x = base_model(preprocessed, training=False)
    
    # 3. Custom Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="transfer_gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="transfer_dense128")(x)
    x = tf.keras.layers.BatchNormalization(name="transfer_bn")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="transfer_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="transfer_sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="VisionGuard_MobileNetV2")
    
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
    return model, base_model

def unfreeze_top_block(model, base_model, fine_tune_lr=1e-5, num_layers_to_unfreeze=30):
    """
    Unfreezes the top convolutional block of MobileNetV2 for fine-tuning.
    Recompiles with a much lower learning rate to protect learned features.
    """
    base_model.trainable = True
    
    # Freeze all layers except the top N layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        # Keep BatchNormalization layers frozen in fine-tuning to prevent mean/var collapse
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            
    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
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

def export_trainable_layers_manifest(model, stage_name, output_path):
    """Exports list of trainable layers for evidence in report."""
    trainable_layers = []
    for layer in model.layers:
        if hasattr(layer, "layers"): # For nested base_model
            for sub_layer in layer.layers:
                if sub_layer.trainable:
                    trainable_layers.append({"Scope": layer.name, "Layer": sub_layer.name, "Type": sub_layer.__class__.__name__})
        else:
            if layer.trainable:
                trainable_layers.append({"Scope": "Head", "Layer": layer.name, "Type": layer.__class__.__name__})
                
    df = pd.DataFrame(trainable_layers)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"\n### {stage_name}: Trainable Layers ({len(trainable_layers)} layers)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"[+] Trainable layers for {stage_name} appended to: {output_path}")

if __name__ == "__main__":
    model, base = build_transfer_model()
    model.summary()

