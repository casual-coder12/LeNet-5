import tensorflow as tf
import keras
from tensorflow.keras import layers, models


def LeNet5(input_shape, num_classes):
    """
    LeNet-5 model implementation using Keras Functional API.
    This approach provides better compatibility with model saving/loading.
    
    Args:
        input_shape: The shape of the input images (without batch dimension).
        num_classes: The number of classes for the classification task.
    
    Returns:
        A compiled Keras functional model.
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')
    
    # C1: Convolutional layer
    x = layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        activation='tanh',
        name='C1_Conv'
    )(inputs)
    
    # S2: Average pooling layer
    x = layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='S2_AvgPool'
    )(x)
    
    # C3: Convolutional layer
    x = layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        activation='tanh',
        name='C3_Conv'
    )(x)
    
    # S4: Average pooling layer
    x = layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='S4_AvgPool'
    )(x)
    
    # C5: Convolutional layer
    x = layers.Conv2D(
        filters=120,
        kernel_size=(5, 5),
        activation='tanh',
        name='C5_Conv'
    )(x)
    
    # Flatten layer
    x = layers.Flatten(name='Flatten')(x)
    
    # F6: Fully connected layer
    x = layers.Dense(
        units=84,
        activation='tanh',
        name='F6_Dense'
    )(x)
    
    # Output layer
    outputs = layers.Dense(
        units=num_classes,
        activation='softmax',
        name='Output_Layer'
    )(x)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='LeNet5')
    
    return model
    

def lenet5_seq(input_shape, num_classes):
    """
    (Alternative) Build the LeNet-5 model using Keras Sequential API.
    Args:
        input_shape: The shape of the input images.
        num_classes: The number of classes for the classification task.
    Returns:
        A compiled Keras model.
    """
    model = models.Sequential(
        [
            layers.Conv2D(
                filters=6, 
                kernel_size=(5, 5), 
                activation='tanh',
                input_shape=input_shape,
                name='C1_Conv'
                ),

            layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name='S2_AvgPool'
                ),
                
            layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                activation='tanh',
                name='C3_Conv'
                ),

            layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name='S4_AvgPool'
                ),

            layers.Conv2D(
                filters=120, 
                kernel_size=(5, 5), 
                activation='tanh',
                name='C5_Conv'
                ),

            layers.Flatten(
                name='Flatten'
            ),

            layers.Dense(
                units=84,
                activation='tanh',
                name='F6_Dense'
                ),

            layers.Dense(
                units=num_classes,
                activation='softmax',
                name='Output_Layer'
                )
        ]
    )
    return model

