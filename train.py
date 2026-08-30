import os
import sys
import argparse

# # Dodavanje korenskog direktorijuma u path
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.lenet5 import LeNet5
from data.dataset import prepare_mnist_dataset, prepare_cifar10_dataset
from utils.trainer import LeNetTrainer
from utils.visualize import plot_training_history

def parse_args():
    parser = argparse.ArgumentParser(description="Train LeNet-5 on MNIST or CIFAR-10 datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to use for training (default: mnist).",
    )
    parser.add_argument(
        "--save_type",
        type=str,
        default="both",
        choices=["w", "m", "both"],
        help="Type of save operation: 'w' for weights, 'm' for entire model, 'both' for both (default: w).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to train (default: 15).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001).",
    )

    return parser.parse_args()

def main():
    """
    Main function to train the LeNet-5 model on the specified dataset (MNIST or CIFAR-10).
    It prepares the dataset, initializes the model and trainer, and starts the training process.
    """

    # # Parameters defined without parse_args()
    # DATASET_NAME = "mnist"  # Options: "mnist" or "cifar10"
    # EPOCHS = 15
    # BATCH_SIZE = 64
    # LEARNING_RATE = 0.001

    args = parse_args()
  
    print(f"=== Starting Training on {args.dataset.upper()} ===")

    # Set input dimensions according to dataset choice
    input_shape = (32, 32, 1) if args.dataset == "mnist" else (32, 32, 3)

    # Get the appropriate dataset
    if args.dataset == "mnist":
        train_data, val_data, test_data = prepare_mnist_dataset(batch_size=args.batch_size)
    elif args.dataset == "cifar10":
        train_data, val_data, test_data = prepare_cifar10_dataset(batch_size=args.batch_size)
    else:
        raise ValueError("Invalid dataset name. Choose 'mnist' or 'cifar10'.")

    # Instantiate model and trainer wrapper
    model = LeNet5(input_shape=input_shape, num_classes=10)
    trainer = LeNetTrainer(model=model, learning_rate=args.learning_rate)

    # Train model
    history = trainer.train(train_data=train_data, val_data=val_data, epochs=args.epochs, dataset_name=args.dataset, save_type=args.save_type)

    plot_training_history(history, dataset_name=args.dataset)

    # Quick evaluation on test data
    results = trainer.evaluate(test_data=test_data, dataset_name=args.dataset)
    print("\n" + "="*50)
    print(f"--- Results on Test Data (for detailed evaluation - run evaluate.py) ---")
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()