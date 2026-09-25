import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics, models

from data.dataset import prepare_mnist_dataset, prepare_cifar10_dataset
from models.lenet5 import LeNet5
from utils.trainer import LeNetTrainer
from utils.visualize import plot_confusion_matrix, plot_sample_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained LeNet-5 model on MNIST or CIFAR-10 datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset name to evaluate on"
    )
    parser.add_argument(
        "--load_type",
        type=str,
        default="w",
        choices=["w", "m"],
        help="Type of load operation: 'w' for weights, 'm' for entire model (default: w).",
        )
    return parser.parse_args()

def evaluate():
    """
    Evaluate the model on the test dataset and print the results.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        test_data (tf.data.Dataset): The test dataset.
    """
    args = parse_args()
    dataset_name = args.dataset

    outputs_dir = "outputs"
    saved_models_dir = "saved_models"
    os.makedirs(outputs_dir, exist_ok=True)
    
    if dataset_name == "mnist":
        _, _, test_data = prepare_mnist_dataset(batch_size=64)
        input_shape = (32, 32, 1)
    elif dataset_name == "cifar10":
        _, _, test_data = prepare_cifar10_dataset(batch_size=64)
        input_shape = (32, 32, 3)
    else:
        raise ValueError("Invalid dataset name. Choose 'mnist' or 'cifar10'.")

    if args.load_type == "w":
        weights_path = os.path.join(saved_models_dir, f"lenet5_{dataset_name}.weights.h5")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at: {weights_path}")
        # Create a new model and load weights
        model = LeNet5(input_shape=input_shape, num_classes=10)
        model.load_weights(weights_path)
        print("Model weights loaded successfully")
    elif args.load_type == "m":
        model_path = os.path.join(saved_models_dir, f"lenet5_{dataset_name}.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Full model file not found at: {model_path}")
        # Load the entire pre-trained model
        model = models.load_model(model_path)
        print("Model loaded successfully")

    trainer = LeNetTrainer(model=model)

    # Evaluate the model on the test dataset
    results = trainer.evaluate(test_data, dataset_name)

    print("\n" + "="*50)
    print(" " * 15 + "Evaluation Results" + " " * 15)
    print("=" * 50 + "\n")
    print(f"Test Loss: {results}")
    print(f"Test Loss: {results["loss"]:.4f}")
    print(f"Test Accuracy: {results["accuracy"]*100:.2f}%")

    plot_confusion_matrix(model, test_data, dataset_name=dataset_name)

    plot_sample_predictions(model, test_data, dataset_name=dataset_name)

if __name__ == "__main__":
    evaluate()