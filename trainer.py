import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import time
from urllib.request import urlretrieve
import tarfile
from tqdm import tqdm
import argparse
from model import *
from data import *
# Trainer class
class Trainer:
    def __init__(self, model, data_loader, batch_size=128, learning_rate_schedule=None,use_augmentation=False,
                checkpoint_dir='checkpoints'):
        """
        Initialize the trainer.
        
        Parameters:
        - model: Neural network model to train
        - data_loader: Data loader for CIFAR-10
        - batch_size: Batch size for training
        - learning_rate_schedule: Dictionary with epochs as keys and learning rates as values
        - checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.learning_rate_schedule = learning_rate_schedule or {}
        self.checkpoint_dir = checkpoint_dir
        self.data_loader.augment_train = use_augmentation
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train(self, num_epochs, print_every=100, validate_every=1):
        """
        Train the neural network.
        
        Parameters:
        - num_epochs: Number of epochs to train
        - print_every: Print training progress every this many iterations
        - validate_every: Perform validation every this many epochs
        
        Returns:
        - Dictionary with training history
        """
        # Initialize tracking variables
        best_val_acc = 0.0
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model_final.pkl')
        print(f"Data augmentation status: {'Enabled' if self.data_loader.augment_train else 'Disabled'}")
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Number of iterations per epoch
        iterations_per_epoch = max(1, self.data_loader.train_data.shape[0] // self.batch_size)
        
        # Start training
        print(f"Training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            # Update learning rate if scheduled
            if epoch in self.learning_rate_schedule:
                self.model.learning_rate = self.learning_rate_schedule[epoch]
                print(f"Epoch {epoch}: Learning rate updated to {self.model.learning_rate}")
            
            # Record current learning rate
            history['learning_rates'].append(self.model.learning_rate)
            history['epochs'].append(epoch)
            
            # Train for one epoch
            epoch_loss = 0.0
            start_time = time.time()
            
            for i in tqdm(range(iterations_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Get batch
                X_batch, y_batch = self.data_loader.get_batch(self.batch_size, 'train')
                
                # Forward pass
                _ = self.model.forward(X_batch)
                
                # Compute loss
                batch_loss = self.model.compute_loss(X_batch, y_batch)
                epoch_loss += batch_loss
                
                # Backward pass
                grads = self.model.backward(y_batch)
                
                # Update parameters
                self.model.update_params(grads)
                
                # Print progress
                if (i+1) % print_every == 0:
                    print(f"  Iteration {i+1}/{iterations_per_epoch}, Batch loss: {batch_loss:.4f}")
            
            # Average epoch loss
            epoch_loss /= iterations_per_epoch
            history['train_loss'].append(epoch_loss)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, Avg loss: {epoch_loss:.4f}")
            
            # Perform validation if needed
            if (epoch+1) % validate_every == 0:
                val_loss, val_acc = self.validate()
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.model.save_model(best_model_path)
                    print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        # Load best model
        self.model.load_model(best_model_path)
        print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return history
    
    def validate(self, batch_size=200):
        """
        Validate the model on the validation set.
        
        Parameters:
        - batch_size: Batch size for validation
        
        Returns:
        - Validation loss and accuracy
        """
        # Number of samples and batches
        num_samples = self.data_loader.val_data.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        total_loss = 0.0
        total_correct = 0
        
        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = self.data_loader.val_data[start_idx:end_idx]
            y_batch = self.data_loader.val_labels[start_idx:end_idx]
            
            # Forward pass
            probs = self.model.forward(X_batch)
            
            # Compute loss
            batch_loss = self.model.compute_loss(X_batch, y_batch)
            total_loss += batch_loss * (end_idx - start_idx)
            
            # Compute accuracy
            predictions = np.argmax(probs, axis=1)
            total_correct += np.sum(predictions == y_batch)
        
        # Compute average loss and accuracy
        avg_loss = total_loss / num_samples
        accuracy = total_correct / num_samples
        
        return avg_loss, accuracy
    
    def test(self, batch_size=200):
        """
        Test the model on the test set.
        
        Parameters:
        - batch_size: Batch size for testing
        
        Returns:
        - Test accuracy
        """
        # Number of samples and batches
        num_samples = self.data_loader.test_data.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        total_correct = 0
        all_predictions = []
        
        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = self.data_loader.test_data[start_idx:end_idx]
            y_batch = self.data_loader.test_labels[start_idx:end_idx]
            
            # Forward pass
            probs = self.model.forward(X_batch)
            
            # Compute predictions
            predictions = np.argmax(probs, axis=1)
            all_predictions.extend(predictions)
            
            # Compute accuracy
            total_correct += np.sum(predictions == y_batch)
        
        # Compute accuracy
        accuracy = total_correct / num_samples
        
        return accuracy, np.array(all_predictions)


# Visualization functions
def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics.
    
    Parameters:
    - history: Dictionary with training history
    - save_path: Path to save the plot (if None, display the plot)
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation accuracy
    if 'val_acc' in history:
        ax2.plot(history['epochs'], history['val_acc'], 'g-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def visualize_weights(model, save_path=None):
    """
    Visualize the weights of the first layer.
    
    Parameters:
    - model: Trained neural network model
    - save_path: Path to save the plot (if None, display the plot)
    """
    # Get weights from the first layer
    W1 = model.params['W1']
    
    # Reshape weights to visualize as images
    # CIFAR-10 has 3 channels and 32x32 images, so we reshape to (3, 32, 32, hidden_size1)
    num_features = W1.shape[0]
    if num_features == 3072:  # 3 * 32 * 32
        weights = W1.T.reshape(-1, 3, 32, 32)
        num_weights = min(25, weights.shape[0])  # Display at most 25 filters
        
        # Create figure
        size = int(np.ceil(np.sqrt(num_weights)))
        fig, axes = plt.subplots(size, size, figsize=(10, 10))
        
        # Plot each filter
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                if idx < num_weights:
                    # Get the filter
                    weight = weights[idx]
                    
                    # Normalize to [0, 1] for visualization
                    weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-9)
                    
                    # Transpose to (32, 32, 3) for plotting
                    weight = weight.transpose(1, 2, 0)
                    
                    axes[i, j].imshow(weight)
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
        
        plt.suptitle('First Layer Weights Visualization', fontsize=16)
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            print(f"Weight visualization saved to {save_path}")
        else:
            plt.show()
    else:
        print("First layer weights don't match expected dimensions for CIFAR-10.")


# Hyperparameter search function
def hyperparameter_search(data_loader, input_size, output_size, hidden_sizes, learning_rates, 
                        reg_lambdas, activations, batch_sizes, num_epochs=10, verbose=False):
    """
    Search for the best hyperparameters.
    
    Parameters:
    - data_loader: Data loader for CIFAR-10
    - input_size: Size of input features
    - output_size: Number of output classes
    - hidden_sizes: List of tuples (hidden_size1, hidden_size2) to try
    - learning_rates: List of learning rates to try
    - reg_lambdas: List of regularization strengths to try
    - activations: List of activation functions to try
    - batch_sizes: List of batch sizes to try
    - num_epochs: Number of epochs to train each model
    - verbose: Whether to print detailed progress
    
    Returns:
    - Dictionary with best hyperparameters and results
    """
    best_val_acc = 0.0
    best_params = {}
    results = []
    
    # Total number of configurations
    total_configs = len(hidden_sizes) * len(learning_rates) * len(reg_lambdas) * len(activations) * len(batch_sizes)
    print(f"Performing hyperparameter search with {total_configs} configurations...")
    
    # Track progress
    config_idx = 0
    
    # Iterate over all hyperparameter combinations
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for reg in reg_lambdas:
                for act in activations:
                    for bs in batch_sizes:
                        config_idx += 1
                        print(f"Configuration {config_idx}/{total_configs}:")
                        print(f"  Hidden sizes: {hidden_size}, LR: {lr}, Reg: {reg}, Activation: {act}, Batch size: {bs}")
                        
                        # Initialize model with current hyperparameters
                        model = ThreeLayerNet(
                            input_size=input_size,
                            hidden_size1=hidden_size[0],
                            hidden_size2=hidden_size[1],
                            output_size=output_size,
                            activation=act,
                            learning_rate=lr,
                            reg_lambda=reg
                        )
                        
                        # Initialize trainer
                        trainer = Trainer(model, data_loader, batch_size=bs)
                        
                        # Train the model for a few epochs
                        if verbose:
                            print("  Training started...")
                        
                        history = trainer.train(num_epochs, print_every=1000, validate_every=1)
                        
                        # Get final validation accuracy
                        val_loss, val_acc = trainer.validate()
                        
                        if verbose:
                            print(f"  Final validation accuracy: {val_acc:.4f}")
                        
                        # Store results
                        result = {
                            'hidden_size': hidden_size,
                            'learning_rate': lr,
                            'reg_lambda': reg,
                            'activation': act,
                            'batch_size': bs,
                            'val_accuracy': val_acc,
                            'val_loss': val_loss,
                            'history' : history,
                        }
                        results.append(result)
                        
                        # Update best parameters if needed
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_params = {
                                'hidden_size': hidden_size,
                                'learning_rate': lr,
                                'reg_lambda': reg,
                                'activation': act,
                                'batch_size': bs,
                                'val_accuracy': val_acc
                            }
                            
                            print(f"  New best validation accuracy: {val_acc:.4f}")
    
    # Print best parameters
    print("\nHyperparameter search completed.")
    print(f"Best validation accuracy: {best_params['val_accuracy']:.4f}")
    print(f"Best parameters:")
    print(f"  Hidden sizes: {best_params['hidden_size']}")
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Regularization: {best_params['reg_lambda']}")
    print(f"  Activation: {best_params['activation']}")
    print(f"  Batch size: {best_params['batch_size']}")
    
    return best_params, results


# Main functions
def train_model(args):
    """
    Train a neural network model with specified parameters.
    
    Parameters:
    - args: Command-line arguments
    """
    # Load data
    data_loader = CIFAR10Loader(data_dir=args.data_dir)
    
    # Initialize model
    model = ThreeLayerNet(
        input_size=3072,  # 3 * 32 * 32
        hidden_size1=args.hidden_size1,
        hidden_size2=args.hidden_size2,
        output_size=10,  # CIFAR-10 has 10 classes
        activation=args.activation,
        learning_rate=args.learning_rate,
        reg_lambda=args.reg_lambda
    )
    
    # Define learning rate schedule if specified
    if args.lr_decay:
        # Learning rate decay - divide by 2 every 10 epochs
        lr_schedule = {}
        for i in range(10, args.num_epochs + 1, 10):
            lr_schedule[i] = args.learning_rate * (0.5 ** (i // 10))
    else:
        lr_schedule = None
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        batch_size=args.batch_size,
        learning_rate_schedule=lr_schedule,
        checkpoint_dir=args.checkpoint_dir,
        use_augmentation=args.use_augmentation,
    )
    
    # Train the model
    history = trainer.train(
        num_epochs=args.num_epochs,
        print_every=args.print_every,
        validate_every=args.validate_every
    )
    
    # Visualize training history
    os.makedirs('figures', exist_ok=True)
    plot_training_history(history, save_path='figures/training_history_final.png')
    
    # Visualize weights
    visualize_weights(model, save_path='figures/weights_visualization_final.png')
    
    # Test the model
    test_acc, _ = trainer.test()
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save final model
    model.save_model(os.path.join(args.checkpoint_dir, 'final_model_final.pkl'))


def find_best_hyperparameters(args):
    """
    Find the best hyperparameters for the model.
    
    Parameters:
    - args: Command-line arguments
    """
    # Load data
    data_loader = CIFAR10Loader(data_dir=args.data_dir)
    
    # Define hyperparameter search space
    hidden_sizes = [
        (128, 64),
        (256, 128),
        (512, 256)
    ]
    learning_rates = [0.01, 0.001, 0.0001]
    reg_lambdas = [0.0, 0.001, 0.01]
    activations = ['relu', 'sigmoid', 'tanh']
    batch_sizes = [64, 128, 256]
    
    # Perform hyperparameter search
    best_params, results = hyperparameter_search(
        data_loader=data_loader,
        input_size=3072,  # 3 * 32 * 32
        output_size=10,   # CIFAR-10 has 10 classes
        hidden_sizes=hidden_sizes,
        learning_rates=learning_rates,
        reg_lambdas=reg_lambdas,
        activations=activations,
        batch_sizes=batch_sizes,
        num_epochs=args.num_epochs_search,
        verbose=args.verbose
    )
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/hyperparameter_search_results.pkl', 'wb') as f:
        pickle.dump({'best_params': best_params, 'all_results': results}, f)
    
    print(f"Hyperparameter search results saved to results/hyperparameter_search_results.pkl")


def test_model(args):
    """
    Test a trained neural network model.
    
    Parameters:
    - args: Command-line arguments
    """
    # Load data
    data_loader = CIFAR10Loader(data_dir=args.data_dir)
    
    # Initialize model (with dummy hyperparameters, will be overwritten by loaded model)
    model = ThreeLayerNet(
        input_size=3072,  # 3 * 32 * 32
        hidden_size1=100,
        hidden_size2=50,
        output_size=10,   # CIFAR-10 has 10 classes
        activation='relu'
    )
    
    # Load trained model
    model.load_model(args.model_path)
    
    # Initialize trainer
    trainer = Trainer(model, data_loader)
    
    # Test the model
    test_acc, predictions = trainer.test()
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save predictions if requested
    if args.save_predictions:
        os.makedirs('results', exist_ok=True)
        np.save('results/test_predictions.npy', predictions)
        print(f"Test predictions saved to results/test_predictions.npy")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate a neural network for CIFAR-10')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train parser
    train_parser = subparsers.add_parser('train', help='Train a neural network')
    train_parser.add_argument('--data-dir', type=str, default='cifar-10-batches-py',
                             help='Directory containing CIFAR-10 data')
    train_parser.add_argument('--hidden-size1', type=int, default=256,
                             help='Size of first hidden layer')
    train_parser.add_argument('--hidden-size2', type=int, default=128,
                             help='Size of second hidden layer')
    train_parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'],
                             help='Activation function')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--reg-lambda', type=float, default=0.001,
                             help='L2 regularization strength')
    train_parser.add_argument('--batch-size', type=int, default=128,
                             help='Batch size')
    train_parser.add_argument('--num-epochs', type=int, default=20,
                             help='Number of epochs')
    train_parser.add_argument('--print-every', type=int, default=100,
                             help='Print progress every this many iterations')
    train_parser.add_argument('--validate-every', type=int, default=1,
                             help='Perform validation every this many epochs')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                             help='Directory to save model checkpoints')
    train_parser.add_argument('--lr-decay', action='store_true',
                             help='Use learning rate decay')
    
    # Search parser
    search_parser = subparsers.add_parser('search', help='Find best hyperparameters')
    search_parser.add_argument('--data-dir', type=str, default='cifar-10-batches-py',
                              help='Directory containing CIFAR-10 data')
    search_parser.add_argument('--num-epochs-search', type=int, default=5,
                              help='Number of epochs for each hyperparameter configuration')
    search_parser.add_argument('--verbose', action='store_true',
                              help='Print detailed progress')
    
    # Test parser
    test_parser = subparsers.add_parser('test', help='Test a trained neural network')
    test_parser.add_argument('--data-dir', type=str, default='cifar-10-batches-py',
                            help='Directory containing CIFAR-10 data')
    test_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to the trained model')
    test_parser.add_argument('--save-predictions', action='store_true',
                            help='Save test predictions')
    
    args = parser.parse_args()
    
    # Run the specified command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'search':
        find_best_hyperparameters(args)
    elif args.command == 'test':
        test_model(args)
    else:
        parser.print_help()