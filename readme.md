# Three-Layer Neural Network for CIFAR-10 Classification

This repository contains an implementation of a three-layer neural network for image classification on the CIFAR-10 dataset. The implementation is done from scratch using only NumPy without relying on deep learning frameworks like PyTorch or TensorFlow.

## Features

- Custom implementation of a three-layer neural network with backpropagation
- Support for different activation functions (ReLU, Sigmoid, Tanh)
- SGD optimizer with learning rate scheduling
- Cross-entropy loss and L2 regularization
- Automatic model checkpointing based on validation metrics
- Hyperparameter tuning framework
- Visualization of training progress and network weights

## Requirements

To run this code, you need the following dependencies:

```
numpy
matplotlib
tqdm
pickle
```

You can install them using pip:

```bash
pip install numpy matplotlib tqdm
```

## Dataset

The code will automatically download and extract the CIFAR-10 dataset if it's not already present in the specified directory.

## Usage

The code provides three main functionalities: training, hyperparameter search, and testing.

### Training a Model

To train a model with default parameters:

```bash
python main.py train
```

You can customize the training with various parameters:

```bash
python main.py train \
    --data-dir cifar-10-batches-py \
    --hidden-size1 256 \
    --hidden-size2 128 \
    --activation relu \
    --learning-rate 0.001 \
    --reg-lambda 0.001 \
    --batch-size 128 \
    --num-epochs 20 \
    --checkpoint-dir checkpoints \
    --lr-decay
```

### Hyperparameter Search

To find the best hyperparameters:

```bash
python main.py search \
    --data-dir cifar-10-batches-py \
    --num-epochs-search 5 \
    --verbose
```

This will explore combinations of hidden layer sizes, learning rates, regularization strengths, activation functions, and batch sizes.

### Testing a Trained Model

To test a trained model:

```bash
python main.py test \
    --data-dir cifar-10-batches-py \
    --model-path checkpoints/best_model.pkl \
    --save-predictions
```

## Model Architecture

The neural network consists of:
- Input layer: 3072 units (3 color channels × 32 × 32 pixels)
- First hidden layer: Configurable size (default 256)
- Second hidden layer: Configurable size (default 128)
- Output layer: 10 units (one for each CIFAR-10 class)

## Files Description

- `main.py`: Contains the complete implementation including the neural network model, data loader, trainer, and utility functions.
- `checkpoints/`: Directory to store model checkpoints
- `figures/`: Directory for visualizations generated during training
- `results/`: Directory for saving hyperparameter search results and test predictions

## Pre-trained Model

A pre-trained model with the best hyperparameters can be downloaded from this link: [Download Pre-trained Model](https://drive.google.com/drive/folders/1mb4THQ786vTa3KIh3nxyIWXoLRNvDNWc?usp=drive_link)

## Results

With the default hyperparameters, the model achieves approximately 50-55% accuracy on the CIFAR-10 test set. The performance can be improved further with proper hyperparameter tuning.

## Acknowledgments

The CIFAR-10 dataset is provided by the Canadian Institute For Advanced Research (CIFAR).

## License

This project is licensed under the MIT License - see the LICENSE file for details.