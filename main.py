#!/usr/bin/env python3
from model import *
from trainer import *
from data import *

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate a neural network for CIFAR-10')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train parser
    train_parser = subparsers.add_parser('train', help='Train a neural network')
    train_parser.add_argument('--data-dir', type=str, default='cifar-10-batches-py',
                             help='Directory containing CIFAR-10 data')
    train_parser.add_argument('--hidden-size1', type=int, default=2048,
                             help='Size of first hidden layer')
    train_parser.add_argument('--hidden-size2', type=int, default=1024,
                             help='Size of second hidden layer')
    train_parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'],
                             help='Activation function')
    train_parser.add_argument('--learning-rate', type=float, default=0.01,
                             help='Learning rate')
    train_parser.add_argument('--reg-lambda', type=float, default=0.01,
                             help='L2 regularization strength')
    train_parser.add_argument('--batch-size', type=int, default=64,
                             help='Batch size')
    train_parser.add_argument('--num-epochs', type=int, default=200,
                             help='Number of epochs')
    train_parser.add_argument('--print-every', type=int, default=100,
                             help='Print progress every this many iterations')
    train_parser.add_argument('--validate-every', type=int, default=1,
                             help='Perform validation every this many epochs')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                             help='Directory to save model checkpoints')
    train_parser.add_argument('--lr-decay', type=bool, default=True,
                             help='Use learning rate decay')
    train_parser.add_argument('--use_augmentation', type=bool, default=False,
                             help='Use augmentation')
    
    # Search parser
    search_parser = subparsers.add_parser('search', help='Find best hyperparameters')
    search_parser.add_argument('--data-dir', type=str, default='cifar-10-batches-py',
                              help='Directory containing CIFAR-10 data')
    search_parser.add_argument('--num-epochs-search', type=int, default=20,
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