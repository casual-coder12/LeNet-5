import sys
import os

# Ensure the root directory is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from tensorflow.keras import datasets, utils
from sklearn.model_selection import train_test_split


def prepare_mnist_dataset(batch_size=32, buffer_size=10000):
    """
    Loads the MNIST dataset and prepares it for training and testing.

    Args:
        batch_size (int): Number of samples per batch.
        buffer_size (int): Buffer size for shuffling the dataset.

    Returns:
        tuple: A tuple containing the training and test datasets.
    """
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    print("MNIST Shapes (Before Preprocessing):", X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # Debugging shapes

    X_train = X_train.astype('float32') / 255.0  # Normalize X_train pixel values to [0, 1]
    X_test = X_test.astype('float32') / 255.0  # Normalize X_test pixel values to [0, 1]

    X_train = X_train.reshape(-1, 28, 28, 1)  # Reshape for MNIST (grayscale)
    X_test = X_test.reshape(-1, 28, 28, 1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    X_train = tf.pad(X_train, [[0, 0], [2, 2], [2, 2], [0, 0]])  # Resize to 32x32 for consistency
    X_val = tf.pad(X_val, [[0, 0], [2, 2], [2, 2], [0, 0]])
    X_test = tf.pad(X_test, [[0, 0], [2, 2], [2, 2], [0, 0]])

    print("MNIST Shapes (After Preprocessing):", X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # Debugging shapes

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset

def prepare_cifar10_dataset(batch_size=32, buffer_size=10000):
    """
    Loads the CIFAR-10 dataset and prepares it for training and testing.

    Args:
        batch_size (int): Number of samples per batch.
        buffer_size (int): Buffer size for shuffling the dataset.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    print("CIFAR-10 Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # Debugging shapes

    X_train = X_train.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    X_test = X_test.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Example usage
    train_ds, val_ds, test_ds = prepare_mnist_dataset(batch_size=64)
    print("MNIST Dataset prepared:")
    print(f"Training batches: {len(train_ds)}")
    print(f"Validation batches: {len(val_ds)}")
    print(f"Test batches: {len(test_ds)}")

    train_ds_cifar, val_ds_cifar, test_ds_cifar = prepare_cifar10_dataset(batch_size=64)
    print("CIFAR-10 Dataset prepared:")
    print(f"Training batches: {len(train_ds_cifar)}")
    print(f"Validation batches: {len(val_ds_cifar)}")
    print(f"Test batches: {len(test_ds_cifar)}")