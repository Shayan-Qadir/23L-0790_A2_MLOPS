import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

def run_toy_experiment(tracking_uri=None, experiment_name="Toy_MLflow_Lab"):
    """
    Trains a tiny Keras model on Fashion-MNIST with 2 distinct learning rates,
    logging parameters, per-epoch metrics, loss curve plot, model, and tags.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # SQLite database backend (MLflow recommended standard)
        local_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(local_uri)
        
    mlflow.set_experiment(experiment_name)
    
    # Load tiny subset of Fashion-MNIST for rapid demonstration
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, y_train = x_train[:2000] / 255.0, y_train[:2000]
    x_val, y_val = x_val[:500] / 255.0, y_val[:500]
    
    learning_rates = [0.01, 0.001]
    runs_info = []
    
    for idx, lr in enumerate(learning_rates):
        run_name = f"toy_run_lr_{lr}"
        print(f"\n[*] Starting MLflow Toy Run #{idx+1}: {run_name}")
        
        with mlflow.start_run(run_name=run_name) as run:
            # 1. Log Tags
            mlflow.set_tag("developer", "Shayan-Qadir")
            mlflow.set_tag("project", "23L0790_A2_Ops")
            mlflow.set_tag("task", "3.1 Tool Lab")
            mlflow.set_tag("dataset", "Fashion-MNIST-Mini")
            
            # 2. Log Hyperparameters
            epochs = 5
            batch_size = 64
            hidden_units = 64
            
            mlflow.log_params({
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_units": hidden_units,
                "optimizer": "Adam",
                "loss_function": "SparseCategoricalCrossentropy"
            })
            
            # 3. Model Architecture
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(hidden_units, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax")
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            
            # Custom callback to log per-epoch metrics to MLflow
            class MLflowEpochLogger(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        for metric_name, val in logs.items():
                            mlflow.log_metric(metric_name, float(val), step=epoch)
                            
            history = model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                callbacks=[MLflowEpochLogger()],
                verbose=1
            )
            
            # 4. Generate & Log Loss / Accuracy Plot
            plot_path = f"toy_curves_lr_{lr}.png"
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(history.history["loss"], label="Train Loss")
            ax[0].plot(history.history["val_loss"], label="Val Loss")
            ax[0].set_title(f"Loss Curve (LR={lr})")
            ax[0].set_xlabel("Epoch")
            ax[0].legend()
            
            ax[1].plot(history.history["accuracy"], label="Train Acc")
            ax[1].plot(history.history["val_accuracy"], label="Val Acc")
            ax[1].set_title(f"Accuracy Curve (LR={lr})")
            ax[1].set_xlabel("Epoch")
            ax[1].legend()
            
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            mlflow.log_artifact(plot_path, artifact_path="figures")
            if os.path.exists(plot_path):
                os.remove(plot_path)
                
            # 5. Log Trained Model
            mlflow.tensorflow.log_model(model, artifact_path="model")
            
            # Final validation summary metrics
            final_val_loss = history.history["val_loss"][-1]
            final_val_acc = history.history["val_accuracy"][-1]
            mlflow.log_metrics({"final_val_loss": final_val_loss, "final_val_acc": final_val_acc})
            
            print(f"[+] Run {run.info.run_id} completed successfully.")
            print(f"    Final Val Loss: {final_val_loss:.4f} | Final Val Accuracy: {final_val_acc:.4f}")
            runs_info.append((run.info.run_id, lr, final_val_loss, final_val_acc))
            
    print("\n" + "=" * 60)
    print(" TOY EXPERIMENT SUMMARY (LOCAL MLFLOW)")
    print("=" * 60)
    for r_id, lr, vl, va in runs_info:
        print(f"Run ID: {r_id[:8]}... | LR: {lr:<6} | Val Loss: {vl:.4f} | Val Acc: {va:.4f}")
    print("=" * 60)
    print("To view local UI, run: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")
    print("=" * 60)

if __name__ == "__main__":
    run_toy_experiment()
