"""
CNN Model Architecture for Digit Recognition
Defines the Convolutional Neural Network structure
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def create_cnn_model():
    """
    Create a CNN model for handwritten digit recognition
    
    Architecture:
    - Conv2D (32 filters, 3x3) + ReLU
    - MaxPooling (2x2)
    - Conv2D (64 filters, 3x3) + ReLU
    - MaxPooling (2x2)
    - Flatten
    - Dense (128) + ReLU
    - Dropout (0.5)
    - Dense (10) + Softmax
    
    Returns:
        model: Compiled Keras model
    """
    print("Building CNN model...")
    
    model = Sequential([
        # First Convolutional Block
        # 32 filters of size 3x3, ReLU activation
        # Input shape: (28, 28, 1) - grayscale images
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Reduce spatial dimensions by half (14x14)
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        # 64 filters of size 3x3, ReLU activation
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        # Reduce spatial dimensions by half (7x7)
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten the 3D output to 1D vector
        Flatten(),
        
        # Fully Connected Layers
        # 128 neurons with ReLU activation
        Dense(128, activation='relu'),
        # Dropout 50% of neurons to prevent overfitting
        Dropout(0.5),
        
        # Output Layer
        # 10 neurons (one for each digit 0-9)
        # Softmax converts to probability distribution
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',                      # Adaptive learning rate optimizer
        loss='categorical_crossentropy',       # Loss for multi-class classification
        metrics=['accuracy']                   # Track accuracy during training
    )
    
    print("Model created successfully!\n")
    
    # Display model architecture
    model.summary()
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_cnn_model()
    print(f"\nTotal parameters: {model.count_params():,}")
