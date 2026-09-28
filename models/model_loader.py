import os
import tensorflow as tf
from keras import models

def load_model(model_class, dataset_name, load_type="m", input_shape=(32, 32, 3), num_classes=10, saved_models_dir="saved_models"):
    """
    Loads a model either by instantiating `model_class` and loading weights (.weights.h5),
    or by loading the entire model architecture + weights (.keras).
    """
    ext = ".weights.h5" if load_type == "w" else ".keras"
    file_path = os.path.join(saved_models_dir, f"lenet5_{dataset_name}{ext}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at: {file_path}")

    if load_type == "w":
        model = model_class(input_shape=input_shape, num_classes=num_classes)
        model.load_weights(file_path)
    else:
        model = models.load_model(file_path)

    print(f"Model successfully loaded from {file_path}")

    return model