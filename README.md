# LeNet-5 Implementation

A comprehensive implementation of the LeNet-5 convolutional neural network for image classification on MNIST and CIFAR-10 datasets.

## Overview

This project implements the classic LeNet-5 architecture, originally developed by Yann LeCun for handwritten digit recognition. The implementation uses TensorFlow/Keras and includes training, evaluation, and visualization utilities.

## Features

- **LeNet-5 Architecture**: Classic CNN model with convolutional and pooling layers
- **Multi-Dataset Support**: Train and evaluate on MNIST and CIFAR-10
- **Flexible Model Management**: Save/load models as weights or complete models
- **Training Visualization**: Real-time training metrics and loss/accuracy plots
- **Evaluation Tools**: Confusion matrix generation and model evaluation metrics
- **Interactive Notebooks**: Jupyter notebook for experimentation and exploration

## Project Structure

```
.
├── data/                          # Data preparation and loading
│   ├── __init__.py
│   └── dataset.py                 # MNIST and CIFAR-10 data preparation
├── models/                        # Model definitions
│   ├── __init__.py
│   └── lenet5.py                  # LeNet-5 model architecture
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── trainer.py                 # Training and evaluation logic
│   └── visualize.py               # Visualization utilities
├── notebooks/                     # Jupyter notebooks
│   └── lenet5.ipynb               # Interactive exploration notebook
├── saved_models/                  # Trained models for evaluation
├── outputs/                       # Generated outputs (plots, confusion matrices)
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LeNet-5
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on MNIST dataset:
```bash
python train.py --dataset mnist --epochs 10
```

Train on CIFAR-10 dataset:
```bash
python train.py --dataset cifar10 --epochs 20
```

Save only weights:
```bash
python train.py --dataset mnist --save_type w
```

Save complete model:
```bash
python train.py --dataset mnist --save_type m
```

Save both:
```bash
python train.py --dataset mnist --save_type both
```

### Evaluation

Evaluate model on test set:
```bash
python evaluate.py mnist w
```

Load entire model and evaluate:
```bash
python evaluate.py cifar10 m
```

### Arguments

**train.py:**
- `--dataset`: Dataset to use ('mnist' or 'cifar10', default: 'mnist')
- `--save_type`: Save format ('w' for weights, 'm' for model, 'both' for both, default: 'w')
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Training batch size (default: 32)

**evaluate.py:**
- `dataset`: Dataset name ('mnist' or 'cifar10')
- `load_type`: Loading format ('w' for weights, 'm' for model)

## Model Architecture

LeNet-5 consists of:
- **Input Layer**: 32×32 grayscale (MNIST) or RGB (CIFAR-10) images
- **Conv Layer 1**: 6 filters, 5×5 kernel
- **Pool Layer 1**: 2×2 average pooling
- **Conv Layer 2**: 16 filters, 5×5 kernel
- **Pool Layer 2**: 2×2 average pooling
- **Flatten Layer**: Convert to 1D vector
- **Dense Layer 1**: 120 units with ReLU
- **Dense Layer 2**: 84 units with ReLU
- **Output Layer**: 10 units with Softmax (10 classes)

## Benchmark Results

| Dataset | Accuracy | Loss |
|---|---|---|
| MNIST | ~99% | ~0.05 |
| CIFAR-10 | ~70–75% | ~0.8–1.0 |

## Outputs

Generated outputs are saved in the `outputs/` directory:
- `lenet5_<dataset>_training_plots.png`: Training loss and accuracy curves
- `lenet5_<dataset>_confusion_matrix.png`: Confusion matrix heatmap

Pre-trained models are saved in `saved_models/`:
- `lenet5_<dataset>.keras`: Complete model (architecture + weights)
- `lenet5_<dataset>.weights.h5`: Weights only
- `lenet5_<dataset>_history.csv`: Training history metrics

## Notebook

Open `notebooks/lenet5.ipynb` for interactive exploration:
```bash
jupyter notebook notebooks/lenet5.ipynb
```

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras API: https://keras.io/

## License

This project is provided for educational purposes.

---

*Neural Networks and Deep Learning — LeNet-5 Project*
