import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training_hist_dict(saved_history: dict, dataset_name: str):

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(saved_history["epoch"], saved_history["loss"], label="Training Loss", color='blue')
    plt.plot(saved_history["epoch"], saved_history["val_loss"], label="Validation Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss over Epochs ({dataset_name} Dataset)")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(saved_history["epoch"], saved_history["accuracy"], label="Training Accuracy", color='green')
    plt.plot(saved_history["epoch"], saved_history["val_accuracy"], label="Validation Accuracy", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training Accuracy over Epochs ({dataset_name} Dataset)")
    plt.legend()

    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    path_to_save_plots = f"outputs/lenet5_{dataset_name}_training_plots.png"

    plt.savefig(path_to_save_plots, dpi=300, bbox_inches='tight')

    plt.show()

def plot_training_history(history: tf.keras.callbacks.History, dataset_name: str):

    hist = history.history
    epochs = range(len(hist['loss']))

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["loss"], label="Training Loss", color='blue')
    plt.plot(epochs, hist["val_loss"], label="Validation Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss over Epochs ({dataset_name} Dataset)")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["accuracy"], label="Training Accuracy", color='green')
    plt.plot(epochs, hist["val_accuracy"], label="Validation Accuracy", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training Accuracy over Epochs ({dataset_name} Dataset)")
    plt.legend()

    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    path_to_save_plots = f"outputs/lenet5_{dataset_name}_training_plots.png"

    plt.savefig(path_to_save_plots, dpi=300, bbox_inches='tight')

    plt.show()

def plot_confusion_matrix(y_true, y_pred, dataset_name = "mnist"):
    """
    Computes and displays a formatted Confusion Matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Define class names based on dataset
    if dataset_name == "mnist":
        class_names = [str(i) for i in range(10)]
    elif dataset_name == "cifar10":
        class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                        "dog", "frog", "horse", "ship", "truck"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({dataset_name} Dataset)')

    os.makedirs("outputs", exist_ok=True)
    path_to_save_cm = f"outputs/lenet5_{dataset_name}_confusion_matrix.png"
    plt.savefig(path_to_save_cm, dpi=300, bbox_inches='tight')

    plt.show()

def plot_sample_predictions():
    return None


if __name__ == "__main__":
    print("--- Testing Standalone Visualization Utils ---")

    # 1. Generate dummy metrics for testing plot_training_history
    dummy_history = {
        "loss": [2.3, 1.5, 0.8, 0.4],
        "val_loss": [2.1, 1.4, 0.9, 0.5],
        "accuracy": [0.2, 0.5, 0.75, 0.88],
        "val_accuracy": [0.25, 0.52, 0.72, 0.85],
        "epoch": [0, 1, 2, 3]
        }

    print("Testing plot_training_history()...")
    plot_training_history(dummy_history, "Dummy")