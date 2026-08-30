import os

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint


class LeNetTrainer:
    """
    Trainer class to manage training, validation, and evaluation processes 
    for the LeNet-5 model using TensorFlow/Keras.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        learning_rate: float = 0.001,
        loss_fn=None,
        optimizer=None,
        ):
        """
        Initializes the trainer with model, optimizer, and loss function.

        Args:
            model (tf.keras.Model): The LeNet-5 model instance to be trained.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            loss_fn: Keras loss function instance. Defaults to CategoricalCrossentropy if None.
            optimizer: Keras optimizer instance. Defaults to Adam if None.
        """
        self.model = model
        self.learning_rate = learning_rate

        # Set default loss function to CategoricalCrossentropy (assuming one-hot encoded labels)
        self.loss_fn = loss_fn or losses.SparseCategoricalCrossentropy()

        # Set default optimizer to Adam
        self.optimizer = optimizer or optimizers.Adam(
            learning_rate=self.learning_rate
        )

        self.metric_fns=[
            metrics.SparseCategoricalAccuracy(name="accuracy"),
            metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
            ]
        
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metric_fns,
        )

    def train(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        dataset_name: str,
        save_type: str,
        epochs: int = 20
        ) -> tf.keras.callbacks.History:
        """
        Trains the LeNet-5 model using tf.data Datasets.

        Args:
            train_data (tf.data.Dataset): Prepared training dataset loader.
            val_data (tf.data.Dataset): Prepared validation dataset loader.
            dataset_name (str): Name of the dataset for saving the model.
            save_type (str): Type of save operation, 'w' for weights, 'm' for entire model, 'both' for both.
            epochs (int): Total number of training epochs. Default is 20.

        Returns:
            tf.keras.callbacks.History: Keras history object containing training metrics.
        """
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
            
        path_to_save_history = os.path.join("saved_models", f"lenet5_{dataset_name}_history.csv")
        path_to_save_checkpoint = os.path.join("saved_models", f"lenet5_{dataset_name}_best_model.keras")

        # Set up CSV logger to log training history to a CSV file
        csv_logger = CSVLogger(path_to_save_history, append=True)

        # Set up ModelCheckpoint to save the best model based on validation loss
        checkpoint = ModelCheckpoint(
            path_to_save_checkpoint, 
            monitor='val_loss', 
            save_best_only=True, 
            mode='min'
            )

        print(f"--- Starting LeNet-5 Training for {epochs} Epochs ---")
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[csv_logger, checkpoint]
        )

        self.save_model(dataset_name=dataset_name, save_type=save_type)

        return self.history
    
    def build(self, input_shape):
        """
        Builds the model by specifying the input shape.

        Args:
            input_shape (tuple): Shape of the input data (excluding batch size).
        """
        self.model.build(input_shape=input_shape)

    def save_model(self, dataset_name: str, save_type: str):
        """
        Saves the model either as weights or the entire model.

        Args:
            dataset_name (str): Name of the dataset for saving the model.
            save_type (str): Type of save operation: 'w' for weights, 'm' for entire model.
        """

        if save_type == "w":
            path_to_save_model = os.path.join("saved_models", f"lenet5_{dataset_name}.weights.h5")
            self.model.save_weights(path_to_save_model)
            print(f"\n=====Model Weights Successfully Saved at {path_to_save_model}=====")
        elif save_type == "m":
            # Save entire model using .h5 format (works well with Functional API)
            path_to_save_model = os.path.join("saved_models", f"lenet5_{dataset_name}.keras")
            self.model.save(path_to_save_model)
            print(f"\n=====Entire Model Successfully Saved at {path_to_save_model}=====")
        elif save_type == "both":
            path_to_save_weights = os.path.join("saved_models", f"lenet5_{dataset_name}.weights.h5")
            path_to_save_model = os.path.join("saved_models", f"lenet5_{dataset_name}.keras")
            self.model.save_weights(path_to_save_weights)
            self.model.save(path_to_save_model)
            print(f"\n=====Model Weights Successfully Saved at {path_to_save_weights}=====")
            print(f"\n=====Entire Model Successfully Saved at {path_to_save_model}=====")
        else:
            raise ValueError("Invalid save_type. Choose 'w', 'm' or 'both'.")

    def evaluate(self, test_data: tf.data.Dataset, dataset_name: str) -> dict:
        """
        Quick Evaluation of the trained model on a test dataset.

        Args:
            test_data (tf.data.Dataset): Prepared test dataset loader.
            dataset_name (str): Name of the dataset for evaluation.

        Returns:
            dict: Evaluation metrics containing test loss and accuracy scores.
        """
        print(f"--- Evaluating LeNet-5 on ({dataset_name})Test Data ---")
        results = self.model.evaluate(test_data, return_dict=True)
        return results

    def predict(self, dataset, max_batches=None):
        """
        Extracts images, true labels, and predicted class probabilities/indices in a SINGLE pass.
        
        Returns:
            images (np.ndarray): Flattened array of input images.
            y_true (np.ndarray): 1D array of true class indices.
            y_pred (np.ndarray): 1D array of predicted class indices.
            raw_predictions (np.ndarray): 2D array of raw prediction probabilities.
        """
        x_list, y_list = [], []

        # Iterating through batches in a SINGLE pass
        for i, (x_batch, y_batch) in enumerate(dataset):
            if max_batches is not None and i >= max_batches:
                break
            x_list.append(x_batch.numpy())
            y_list.append(y_batch.numpy())

        images = np.concatenate(x_list, axis=0)
        y_true_raw = np.concatenate(y_list, axis=0)

        # Making predictions on collected images
        raw_predictions = self.model.predict(images, verbose=0)

        # Converting labels and predictions to 1D indices (0-9)
        y_true = np.argmax(y_true_raw, axis=1) if y_true_raw.ndim > 1 else y_true_raw
        y_pred = np.argmax(raw_predictions, axis=1)

        return images, y_true, y_pred, raw_predictions

    def load_model_weights(self, weights_path, input_shape):
        """Builds the architecture layers first, then loads weights."""
        self.model.build((None,) + input_shape)
        self.model.load_weights(weights_path)
        print(f"[INFO] Successfully loaded weights from {weights_path}")


if __name__ == "__main__":
    import numpy as np
    from models.lenet5 import LeNet5

    # Mock dataset creation for code verification (100 samples of 32x32x1 grayscale images)
    dummy_x = np.random.randn(100, 32, 32, 1).astype(np.float32)
    dummy_y = tf.keras.utils.to_categorical(np.random.randint(0, 10, size=(100,)), num_classes=10)

    dataset = tf.data.Dataset.from_tensor_slices((dummy_x, dummy_y)).batch(32)

    # Initialize model and trainer
    lenet = LeNet5(input_shape=(32, 32, 1), num_classes=10)
    trainer = LeNetTrainer(model=lenet, learning_rate=0.001)

    # Run quick test training for 1 epoch
    history = trainer.train(train_data=dataset, val_data=dataset, dataset_name="mnist", save_type="w", epochs=1)
    eval_metrics = trainer.evaluate(test_data=dataset)

    print("\nVerification Test Metrics:", eval_metrics)