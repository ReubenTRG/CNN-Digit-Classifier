"""
Dataset Loader for MNIST Handwritten Digits
Loads and preprocesses the MNIST dataset for training and testing
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_data():
    """
    Load and preprocess the MNIST dataset
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
               All images normalized to [0,1] range
               Labels converted to one-hot encoding
    """
    print("Loading MNIST dataset...")
    
    # Load the dataset (60,000 training images, 10,000 test images)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]} (28x28 grayscale)")
    
    # Normalize pixel values from [0, 255] to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape images to include channel dimension (28, 28, 1)
    # CNN expects (batch_size, height, width, channels)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Convert labels to one-hot encoding
    # Example: 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"Preprocessed shape: {X_train.shape}")
    print(f"Label shape: {y_train.shape}")
    print("Dataset loaded and preprocessed successfully!\n")
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Test the data loader
    X_train, y_train, X_test, y_test = load_data()
    print(f"Sample image pixel range: [{X_train.min()}, {X_train.max()}]")
