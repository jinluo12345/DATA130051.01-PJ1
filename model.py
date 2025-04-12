import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import time
from urllib.request import urlretrieve
import tarfile
from tqdm import tqdm
import argparse

# Define activation functions and their derivatives
class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def softmax(x):
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

# Loss functions
class Loss:
    @staticmethod
    def cross_entropy(y_pred, y_true):
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # Convert y_true to one-hot if not already
        if len(y_true.shape) == 1:
            num_samples = y_true.shape[0]
            num_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((num_samples, num_classes))
            y_true_one_hot[np.arange(num_samples), y_true.astype(int)] = 1
            y_true = y_true_one_hot
        
        # Calculate cross entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        # Convert y_true to one-hot if not already
        if len(y_true.shape) == 1:
            num_samples = y_true.shape[0]
            num_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((num_samples, num_classes))
            y_true_one_hot[np.arange(num_samples), y_true.astype(int)] = 1
            y_true = y_true_one_hot
        
        # Derivative of cross entropy with respect to y_pred
        return (y_pred - y_true) / y_true.shape[0]


# Neural Network Model
class ThreeLayerNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, 
                 activation='relu', learning_rate=0.001, reg_lambda=0.001):
        """
        Initialize the neural network with specified layer sizes and activation function.
        
        Parameters:
        - input_size: Number of input features
        - hidden_size1: Size of first hidden layer
        - hidden_size2: Size of second hidden layer
        - output_size: Number of output classes
        - activation: Activation function to use ('relu', 'sigmoid', or 'tanh')
        - learning_rate: Initial learning rate
        - reg_lambda: L2 regularization strength
        """
        # Set activation function
        if activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        else:
            raise ValueError("Activation function must be 'relu', 'sigmoid', or 'tanh'")
        
        # Initialize parameters
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        # Initialize weights and biases with He initialization
        self.params = {
            'W1': np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size),
            'b1': np.zeros(hidden_size1),
            'W2': np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1),
            'b2': np.zeros(hidden_size2),
            'W3': np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2),
            'b3': np.zeros(output_size)
        }
        
        # Initialize cache for storing intermediate values during forward pass
        self.cache = {}
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        - X: Input data of shape (batch_size, input_size)
        
        Returns:
        - Probability distribution over classes
        """
        # First hidden layer
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = self.activation(z1)
        
        # Second hidden layer
        z2 = a1.dot(self.params['W2']) + self.params['b2']
        a2 = self.activation(z2)
        
        # Output layer (no activation yet)
        z3 = a2.dot(self.params['W3']) + self.params['b3']
        
        # Apply softmax
        out = Activation.softmax(z3)
        
        # Cache values for backward pass
        self.cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2,
            'z3': z3,
            'out': out
        }
        
        return out
    
    def backward(self, y):
        """
        Backward pass to compute gradients.
        
        Parameters:
        - y: True labels (batch_size, output_size) or class indices (batch_size,)
        
        Returns:
        - Dictionary of gradients
        """
        # Get batch size
        batch_size = self.cache['X'].shape[0]
        
        # Convert y to one-hot if needed
        if len(y.shape) == 1:
            num_classes = self.params['W3'].shape[1]
            y_one_hot = np.zeros((batch_size, num_classes))
            y_one_hot[np.arange(batch_size), y.astype(int)] = 1
            y = y_one_hot
        
        # Backpropagation
        # Output layer
        dz3 = self.cache['out'] - y  # This is the derivative of softmax + cross-entropy
        dW3 = self.cache['a2'].T.dot(dz3) / batch_size + self.reg_lambda * self.params['W3']
        db3 = np.sum(dz3, axis=0) / batch_size
        
        # Second hidden layer
        da2 = dz3.dot(self.params['W3'].T)
        dz2 = da2 * self.activation_derivative(self.cache['z2'])
        dW2 = self.cache['a1'].T.dot(dz2) / batch_size + self.reg_lambda * self.params['W2']
        db2 = np.sum(dz2, axis=0) / batch_size
        
        # First hidden layer
        da1 = dz2.dot(self.params['W2'].T)
        dz1 = da1 * self.activation_derivative(self.cache['z1'])
        dW1 = self.cache['X'].T.dot(dz1) / batch_size + self.reg_lambda * self.params['W1']
        db1 = np.sum(dz1, axis=0) / batch_size
        
        # Store gradients
        grads = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2,
            'W3': dW3,
            'b3': db3
        }
        
        return grads
    
    def update_params(self, grads, learning_rate=None):
        """
        Update network parameters using computed gradients.
        
        Parameters:
        - grads: Dictionary of gradients
        - learning_rate: Learning rate to use (if None, use the stored learning rate)
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # Update each parameter
        for param in self.params:
            self.params[param] -= learning_rate * grads[param]
    
    def predict(self, X):
        """
        Make predictions for input data.
        
        Parameters:
        - X: Input data
        
        Returns:
        - Predicted class indices
        """
        # Forward pass
        probs = self.forward(X)
        
        # Return class with highest probability
        return np.argmax(probs, axis=1)
    
    def compute_loss(self, X, y, include_regularization=True):
        """
        Compute the cross-entropy loss and optionally add regularization.
        
        Parameters:
        - X: Input data
        - y: True labels
        - include_regularization: Whether to include L2 regularization
        
        Returns:
        - Total loss value
        """
        # Get outputs
        probs = self.forward(X)
        
        # Compute cross-entropy loss
        data_loss = Loss.cross_entropy(probs, y)
        
        # Add regularization if requested
        if include_regularization:
            reg_loss = 0.5 * self.reg_lambda * (
                np.sum(self.params['W1'] * self.params['W1']) +
                np.sum(self.params['W2'] * self.params['W2']) +
                np.sum(self.params['W3'] * self.params['W3'])
            )
            return data_loss + reg_loss
        else:
            return data_loss
    
    def save_model(self, filename):
        """
        Save model parameters to file.
        
        Parameters:
        - filename: Path to save the model
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load model parameters from file.
        
        Parameters:
        - filename: Path to the saved model
        """
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
        print(f"Model loaded from {filename}")


