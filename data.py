import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import time
from urllib.request import urlretrieve
import tarfile
from tqdm import tqdm
import argparse
from scipy.ndimage import rotate
# CIFAR-10 Data Loader
class CIFAR10Loader:
    def __init__(self, data_dir='cifar-10-batches-py', validation_ratio=0.1,augment_train=True):
        """
        Initialize CIFAR-10 data loader.
        
        Parameters:
        - data_dir: Directory containing CIFAR-10 data
        - validation_ratio: Portion of training data to use for validation
        """
        self.data_dir = data_dir
        self.validation_ratio = validation_ratio
        self.augment_train = augment_train
        # Download CIFAR-10 if needed
        self.maybe_download_and_extract()
        
        # Load data
        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_data()
        
        # Normalize data
        self.preprocess_data()
        
        print(f"Data loaded with shapes:")
        print(f"  Train: {self.train_data.shape}, {self.train_labels.shape}")
        print(f"  Validation: {self.val_data.shape}, {self.val_labels.shape}")
        print(f"  Test: {self.test_data.shape}, {self.test_labels.shape}")

    def augment_batch(self, X_batch):
        """
        对一批数据进行增强处理，添加旋转和颜色反转
        Args:
            X_batch: 输入数据 (batch_size, 3072)
        Returns:
            增强后的数据 (batch_size, 3072)
        """
        X_denorm = X_batch * self.std + self.mean
        augmented_X = []
        
        for x in X_denorm:
            img = x.reshape(3, 32, 32).transpose(1, 2, 0)

            if np.random.rand() < 0.2:
                img = img[:, ::-1, :]

            if np.random.rand() < 0.2:
                angle = np.random.uniform(-60, 60)
                img = rotate(img, angle, reshape=False, mode='reflect')
                
            pad = 4
            img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
            h_start = np.random.randint(0, 2*pad)
            w_start = np.random.randint(0, 2*pad)
            img = img_padded[h_start:h_start+32, w_start:w_start+32, :]
            
            img = img.transpose(2, 0, 1).reshape(-1)
            img = (img - self.mean) / self.std 
            augmented_X.append(img)
            
        return np.array(augmented_X)

    def maybe_download_and_extract(self):
        """Download and extract CIFAR-10 dataset if not already present."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            filename = os.path.join(os.path.dirname(self.data_dir), 'cifar-10-python.tar.gz')
            
            # Download if file doesn't exist
            if not os.path.exists(filename):
                print("Downloading CIFAR-10 dataset...")
                urlretrieve(url, filename)
                print("Download complete")
            
            # Extract
            print("Extracting...")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(os.path.dirname(self.data_dir))
            print("Extraction complete")
    
    def load_batch(self, batch_file):
        """Load a single batch of CIFAR-10 data."""
        with open(batch_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            X = data_dict[b'data']
            Y = data_dict[b'labels']
            return X, np.array(Y)
    
    def load_data(self):
        """Load all CIFAR-10 data."""
        # Load training data
        X_train = []
        y_train = []
        
        for i in range(1, 6):
            batch_file = os.path.join(self.data_dir, f'data_batch_{i}')
            X_batch, y_batch = self.load_batch(batch_file)
            X_train.append(X_batch)
            y_train.append(y_batch)
        
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        
        # Load test data
        test_batch = os.path.join(self.data_dir, 'test_batch')
        X_test, y_test = self.load_batch(test_batch)
        
        # Split training into train and validation
        num_val = int(X_train.shape[0] * self.validation_ratio)
        np.random.seed(42)
        indices = np.random.permutation(X_train.shape[0])
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def preprocess_data(self):
        """Preprocess data: reshape and normalize."""
        # Reshape data: (N, 3, 32, 32) -> (N, 3*32*32)
        self.train_data = self.train_data.reshape(self.train_data.shape[0], -1)
        self.val_data = self.val_data.reshape(self.val_data.shape[0], -1)
        self.test_data = self.test_data.reshape(self.test_data.shape[0], -1)
        
        # Normalize data to [0, 1]
        self.train_data = self.train_data.astype(np.float32) / 255.0
        self.val_data = self.val_data.astype(np.float32) / 255.0
        self.test_data = self.test_data.astype(np.float32) / 255.0
        
        # Precompute mean and std from training data
        self.mean = np.mean(self.train_data, axis=0)
        self.std = np.std(self.train_data, axis=0) + 1e-8  # Add small value to avoid division by zero
        
        # Standardize to zero mean and unit variance
        self.train_data = (self.train_data - self.mean) / self.std
        self.val_data = (self.val_data - self.mean) / self.std
        self.test_data = (self.test_data - self.mean) / self.std
    
    def get_batch(self, batch_size, data_type='train', force_no_augment=False):
        """
        Get a random batch of data with augmentation control
        """
        if data_type == 'train':
            data, labels = self.train_data, self.train_labels
            apply_augment = self.augment_train and not force_no_augment
        elif data_type == 'val':
            data, labels = self.val_data, self.val_labels
            apply_augment = False
        elif data_type == 'test':
            data, labels = self.test_data, self.test_labels
            apply_augment = False
        else:
            raise ValueError("data_type must be 'train', 'val', or 'test'")
        
        idx = np.random.choice(data.shape[0], batch_size, replace=False)
        batch_data = data[idx]
        
        if apply_augment:
            batch_data = self.augment_batch(batch_data)
            
        return batch_data, labels[idx]

def visualize_augmentation(data_loader, num_samples=3, num_augments=2, save_path=None):
    """
    Fixed 3x3 visualization with guaranteed original samples
    """
    # Get original samples (force no augmentation)
    orig_images, labels = data_loader.get_batch(num_samples, 'train', force_no_augment=True)
    
    # Generate multiple augmented versions
    augmentations = [data_loader.augment_batch(orig_images.copy()) for _ in range(num_augments)]
    
    # Prepare visualization
    fig, axs = plt.subplots(num_samples, num_augments+1, figsize=(12, 4*num_samples))
    mean_img = data_loader.mean.reshape(3, 32, 32).transpose(1, 2, 0)
    std_img = data_loader.std.reshape(3, 32, 32).transpose(1, 2, 0)
    
    for i in range(num_samples):
        # Original image (first column)
        img = orig_images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        denorm_img = np.clip(img * std_img + mean_img, 0, 1)
        axs[i,0].imshow(denorm_img)
        axs[i,0].axis('off')
        if i == 0:
            axs[i,0].set_title("Original", fontsize=10)
        
        # Augmented versions (subsequent columns)
        for j in range(num_augments):
            aug_img = augmentations[j][i].reshape(3, 32, 32).transpose(1, 2, 0)
            denorm_aug = np.clip(aug_img * std_img + mean_img, 0, 1)
            axs[i,j+1].imshow(denorm_aug)
            axs[i,j+1].axis('off')
            if i == 0:
                axs[i,j+1].set_title(f"Aug {j+1}", fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle("Data Augmentation Visualization (Original vs Augmented)", y=0.95, fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    data_loader = CIFAR10Loader(augment_train=True)
    visualize_augmentation(
        data_loader,
        num_samples=3,
        num_augments=2,
        save_path='3x3_augmentation_grid.png'
    )