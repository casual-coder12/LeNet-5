import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

CLASS_NAMES = {
    "mnist": [str(i) for i in range(10)],
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
}

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


def plot_training_hist_dict(saved_history: pd.DataFrame, dataset_name: str):

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
    

def plot_confusion_matrix(model, dataset, dataset_name = "mnist"):
    """
    Computes and displays a formatted Confusion Matrix heatmap.
    """
    # Extract true labels from test_data and convert predictions to class indices
    y_true = np.concatenate([y for x, y in dataset], axis=0)

    # Make predictions
    y_pred = np.argmax(model.predict(dataset), axis=1)

    cm = confusion_matrix(y_true, y_pred)

    # Define class names based on dataset
    class_names = CLASS_NAMES[dataset_name]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({dataset_name} Dataset)')

    os.makedirs("outputs", exist_ok=True)
    path_to_save_cm = f"outputs/lenet5_{dataset_name}_confusion_matrix.png"
    plt.savefig(path_to_save_cm, dpi=300, bbox_inches='tight')

    plt.show()

def plot_sample_predictions(model, dataset, dataset_name="mnist", num_samples=10):
    """
    Plots a grid of sample images with their true and predicted labels.

    Args:
        model (tf.keras.Model): The trained model used for predictions.
        dataset (tf.data.Dataset): The dataset containing images and labels.
        dataset_name (str): Name of the dataset ("mnist" or "cifar10").
        num_samples (int): Number of samples to display.
    """
    class_names = CLASS_NAMES[dataset_name]

    # Unbatch, shuffle, and take a subset of the dataset for visualization
    sample_ds = dataset.unbatch().shuffle(buffer_size=10000).take(num_samples)
    x_samples, y_true = next(iter(sample_ds.batch(num_samples)))

    # Make predictions
    y_pred = np.argmax(model.predict(x_samples), axis=1)

    num_samples = min(num_samples, len(x_samples))

    plt.figure(figsize=(15, 5))
    
    for i in range(num_samples):
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(x_samples[i], cmap='gray' if dataset_name == "mnist" else None)
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        is_correct = y_true[i] == y_pred[i]
        title_color = "green" if is_correct else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    path_to_save_samples = f"outputs/lenet5_{dataset_name}_sample_predictions.png"
    plt.savefig(path_to_save_samples, dpi=300, bbox_inches='tight')

    plt.show()


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