"""
Training Script for Digit Recognition CNN
Trains the model on MNIST dataset and saves it
"""

import matplotlib.pyplot as plt
from model import create_cnn_model
from dataset_loader import load_data


def plot_training_history(history):
    """
    Plot training accuracy and loss over epochs
    
    Args:
        history: Keras training history object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training plot saved as 'training_history.png'")
    plt.show()


def train_model():
    """
    Main training function
    Loads data, creates model, trains it, and saves the trained model
    """
    print("="*60)
    print("HANDWRITTEN DIGIT RECOGNITION - TRAINING")
    print("="*60 + "\n")
    
    # Step 1: Load and preprocess data
    X_train, y_train, X_test, y_test = load_data()
    
    # Step 2: Create the CNN model
    model = create_cnn_model()
    
    # Step 3: Train the model
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,              # Process 32 images at a time
        epochs=10,                  # Train for 10 complete passes through data
        validation_data=(X_test, y_test),  # Evaluate on test set after each epoch
        verbose=1                   # Show progress bar
    )
    
    # Step 4: Evaluate the model
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60 + "\n")
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # Step 5: Save the trained model
    model_path = 'digit_cnn.h5'
    model.save(model_path)
    print(f"\nModel saved successfully as '{model_path}'")
    
    # Step 6: Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now run 'predict_draw.py' to test the model with drawings!")


if __name__ == "__main__":
    train_model()
