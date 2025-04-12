import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from model import ThreeLayerNet
from data import CIFAR10Loader
from trainer import Trainer

def run_ablation_study(ablation_type, values_to_test, fixed_params, num_epochs=20, 
                      data_dir='cifar-10-batches-py', checkpoint_dir='ablation_checkpoints',
                      plot_path='figures/ablation_study.png'):
    """
    Run an ablation study by varying one parameter and fixing others.
    
    Parameters:
    - ablation_type: Parameter to vary ('hidden_sizes', 'hidden_size1', 'hidden_size2', 'learning_rate', 
                    'reg_lambda', 'batch_size', 'activation')
    - values_to_test: List of values to test for the ablation parameter
                     For 'hidden_sizes', this should be a list of tuples: [(h1, h2), (h1, h2), ...]
    - fixed_params: Dictionary of fixed parameters
    - num_epochs: Number of epochs to train each model
    - data_dir: Directory containing CIFAR-10 data
    - checkpoint_dir: Directory to save model checkpoints
    - plot_path: Path to save the ablation plot
    """
    # Load data
    data_loader = CIFAR10Loader(data_dir=data_dir, augment_train=fixed_params.get('use_augmentation', False))
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # Prepare for results collection
    train_losses_all = []
    val_accs_all = []
    labels = []
    
    print(f"Running ablation study for parameter: {ablation_type}")
    print(f"Testing values: {values_to_test}")
    
    # Run experiment for each value
    for value in values_to_test:
        # Create model parameters based on fixed parameters, but with the ablation parameter varied
        model_params = fixed_params.copy()
        
        # Special handling for different parameter types
        if ablation_type == 'batch_size':
            batch_size = value
            model_params.pop('batch_size', None)  # Remove from model params if present
            label = f"batch_size={value}"
        elif ablation_type == 'hidden_sizes':
            # Value is a tuple of (hidden_size1, hidden_size2)
            hidden_size1, hidden_size2 = value
            model_params['hidden_size1'] = hidden_size1
            model_params['hidden_size2'] = hidden_size2
            batch_size = model_params.pop('batch_size', 128)  # Default if not specified
            label = f"hidden=({hidden_size1},{hidden_size2})"
        else:
            batch_size = model_params.pop('batch_size', 128)  # Default if not specified
            model_params[ablation_type] = value
            label = f"{ablation_type}={value}"
            
        # Add label for the plot
        labels.append(label)
        
        print(f"\nTesting {label}")
        print(f"Parameters: {model_params}")
        
        # Initialize model
        model = ThreeLayerNet(
            input_size=3072,  # 3 * 32 * 32
            hidden_size1=model_params.get('hidden_size1', 256),
            hidden_size2=model_params.get('hidden_size2', 128),
            output_size=10,  # CIFAR-10 has 10 classes
            activation=model_params.get('activation', 'relu'),
            learning_rate=model_params.get('learning_rate', 0.001),
            reg_lambda=model_params.get('reg_lambda', 0.001)
        )
        
        # Define learning rate schedule if specified
        if model_params.get('lr_decay', False):
            # Learning rate decay - divide by 2 every 10 epochs
            lr_schedule = {}
            lr = model_params.get('learning_rate', 0.001)
            for i in range(10, num_epochs + 1, 10):
                lr_schedule[i] = lr * (0.5 ** (i // 10))
        else:
            lr_schedule = None
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            data_loader=data_loader,
            batch_size=batch_size,
            learning_rate_schedule=lr_schedule,
            checkpoint_dir=os.path.join(checkpoint_dir, f"{ablation_type}_{value}"),
            use_augmentation=model_params.get('use_augmentation', False)
        )
        
        # Train the model
        history = trainer.train(
            num_epochs=num_epochs,
            print_every=num_epochs * 10,  # Less frequent printing
            validate_every=1  # Validate every epoch
        )
        
        # Save results
        train_losses_all.append(history['train_loss'])
        val_accs_all.append(history['val_acc'])
        
        # Save model and history
        model_path = os.path.join(checkpoint_dir, f"{ablation_type}_{value}", "final_model.pkl")
        history_path = os.path.join(checkpoint_dir, f"{ablation_type}_{value}", "history.pkl")
        model.save_model(model_path)
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        # Test the model
        test_acc, _ = trainer.test()
        print(f"Test accuracy for {label}: {test_acc:.4f}")
    
    # Create visualization
    visualize_ablation_results(
        ablation_type=ablation_type,
        values=values_to_test,
        train_losses=train_losses_all,
        val_accs=val_accs_all,
        epochs=history['epochs'],
        labels=labels,
        save_path=plot_path
    )
    
    return train_losses_all, val_accs_all

def visualize_ablation_results(ablation_type, values, train_losses, val_accs, epochs, labels, save_path=None):
    """
    Visualize results from an ablation study.
    
    Parameters:
    - ablation_type: Parameter that was varied
    - values: List of values that were tested
    - train_losses: List of lists containing training losses for each value
    - val_accs: List of lists containing validation accuracies for each value
    - epochs: List of epoch numbers
    - labels: Labels for the legend
    - save_path: Path to save the figure
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Use a color cycle
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(values)))
    
    # Plot training loss
    for i, loss_history in enumerate(train_losses):
        ax1.plot(epochs, loss_history, label=labels[i], color=colors[i], marker='o', markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'Training Loss vs Epochs for Different {ablation_type} Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    for i, acc_history in enumerate(val_accs):
        ax2.plot(epochs, acc_history, label=labels[i], color=colors[i], marker='o', markersize=4)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title(f'Validation Accuracy vs Epochs for Different {ablation_type} Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle(f'Ablation Study: Effect of {ablation_type} on Model Performance', fontsize=16, y=1.05)
    
    # Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Ablation study visualization saved to {save_path}")
    else:
        plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run ablation study for neural network on CIFAR-10')
    parser.add_argument('--ablation-type', type=str, required=True,
                       choices=['hidden_sizes', 'hidden_size1', 'hidden_size2', 'learning_rate', 'reg_lambda', 'batch_size', 'activation'],
                       help='Parameter to vary in the ablation study')
    parser.add_argument('--values', type=str, required=True,
                       help='Comma-separated list of values to test (e.g., "64,128,256" for hidden sizes, or "128_64,256_128,512_256" for hidden_sizes)')
    parser.add_argument('--data-dir', type=str, default='cifar-10-batches-py',
                       help='Directory containing CIFAR-10 data')
    parser.add_argument('--num-epochs', type=int, default=2,
                       help='Number of epochs to train each model')
    parser.add_argument('--hidden-size1', type=int, default=512,
                       help='Size of first hidden layer (fixed parameter)')
    parser.add_argument('--hidden-size2', type=int, default=256,
                       help='Size of second hidden layer (fixed parameter)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate (fixed parameter)')
    parser.add_argument('--reg-lambda', type=float, default=0.01,
                       help='L2 regularization strength (fixed parameter)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (fixed parameter)')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'sigmoid', 'tanh'],
                       help='Activation function (fixed parameter)')
    parser.add_argument('--use-augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--lr-decay', action='store_true',
                       help='Use learning rate decay')
    parser.add_argument('--checkpoint-dir', type=str, default='ablation_checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--plot-path', type=str, default='figures/ablation_study.png',
                       help='Path to save the ablation plot')
    
    args = parser.parse_args()
    
    # Convert values to appropriate type based on ablation_type
    if args.ablation_type == 'hidden_sizes':
        # Parse pairs like "128_64,256_128,512_256"
        values_to_test = []
        for pair in args.values.split(','):
            if '_' not in pair:
                raise ValueError(f"For hidden_sizes, values must be in format 'h1_h2,h1_h2,...' but got {pair}")
            h1, h2 = pair.split('_')
            values_to_test.append((int(h1), int(h2)))
    elif args.ablation_type in ['hidden_size1', 'hidden_size2', 'batch_size']:
        values_to_test = [int(val) for val in args.values.split(',')]
    elif args.ablation_type in ['learning_rate', 'reg_lambda']:
        values_to_test = [float(val) for val in args.values.split(',')]
    elif args.ablation_type == 'activation':
        values_to_test = args.values.split(',')
        # Validate activation functions
        for act in values_to_test:
            if act not in ['relu', 'sigmoid', 'tanh']:
                raise ValueError(f"Invalid activation function: {act}")
    else:
        raise ValueError(f"Unknown ablation type: {args.ablation_type}")
    
    # Set fixed parameters
    fixed_params = {
        'hidden_size1': args.hidden_size1,
        'hidden_size2': args.hidden_size2,
        'learning_rate': args.learning_rate,
        'reg_lambda': args.reg_lambda,
        'batch_size': args.batch_size,
        'activation': args.activation,
        'use_augmentation': args.use_augmentation,
        'lr_decay': args.lr_decay
    }
    
    # Run ablation study
    run_ablation_study(
        ablation_type=args.ablation_type,
        values_to_test=values_to_test,
        fixed_params=fixed_params,
        num_epochs=args.num_epochs,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        plot_path=args.plot_path
    )

if __name__ == "__main__":
    main()