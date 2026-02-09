# 🎨 Handwritten Digit Recognition with CNN

A complete beginner-friendly Python project that uses Convolutional Neural Networks (CNN) to recognize handwritten digits. Features an **interactive drawing canvas** where you can draw digits with your mouse and get real-time predictions!

## 📋 Features

- ✅ Train a CNN model on the MNIST dataset
- ✅ Achieve ~99% accuracy on digit recognition
- ✅ Interactive drawing canvas using Tkinter
- ✅ Real-time digit prediction
- ✅ Clean, modular, and well-documented code
- ✅ Beginner-friendly implementation

## 🖼️ Project Structure

```
digit_recognition/
├── dataset_loader.py      # Load and preprocess MNIST data
├── model.py              # CNN architecture definition
├── train.py              # Training script
├── predict_draw.py       # Interactive drawing canvas + prediction
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Installation

### Step 1: Install Python
Make sure you have Python 3.8 or higher installed.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow/Keras (Deep Learning framework)
- NumPy (Numerical computing)
- Matplotlib (Plotting)
- OpenCV (Image processing)
- Pillow (Image handling)

## 🚀 Usage

### Step 1: Train the Model

Run the training script to train the CNN on the MNIST dataset:

```bash
python train.py
```

This will:
- Download the MNIST dataset automatically (first time only)
- Train the CNN for 10 epochs
- Display training progress
- Save the trained model as `digit_cnn.h5`
- Generate a training history plot (`training_history.png`)
- Show final test accuracy (~99%)

**Training takes about 2-5 minutes on a modern CPU.**

### Step 2: Test with Drawing Canvas

After training, launch the interactive canvas:

```bash
python predict_draw.py
```

This opens a GUI where you can:
1. Draw a digit (0-9) using your mouse
2. Click **Predict** to see what the model thinks it is
3. Click **Clear** to draw a new digit

## 🧠 CNN Architecture

```
Input (28x28x1)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling (2x2)
    ↓
Flatten
    ↓
Dense (128) + ReLU
    ↓
Dropout (50%)
    ↓
Dense (10) + Softmax
    ↓
Output (10 classes: 0-9)
```

## 📊 Model Performance

- **Training Samples**: 60,000 images
- **Test Samples**: 10,000 images
- **Expected Accuracy**: ~99%
- **Training Time**: 2-5 minutes (CPU)

## 🎯 How It Works

### Training Phase (`train.py`)

1. **Load Data**: Downloads MNIST dataset (28×28 grayscale images)
2. **Normalize**: Converts pixel values from [0, 255] to [0, 1]
3. **Build Model**: Creates CNN with 2 convolutional blocks
4. **Train**: Uses Adam optimizer for 10 epochs
5. **Evaluate**: Tests on 10,000 unseen images
6. **Save**: Stores model as `digit_cnn.h5`

### Prediction Phase (`predict_draw.py`)

1. **Draw**: User draws digit on 400×400 black canvas
2. **Preprocess**:
   - Convert to grayscale
   - Find digit bounding box
   - Center and resize to 28×28
   - Normalize to [0, 1]
3. **Predict**: Feed to CNN model
4. **Display**: Show predicted digit with confidence

## 🔍 Code Explanation

### `dataset_loader.py`
Handles MNIST data loading and preprocessing:
- Normalizes images to [0, 1] range
- Reshapes for CNN input (28, 28, 1)
- Converts labels to one-hot encoding

### `model.py`
Defines the CNN architecture:
- 2 convolutional blocks for feature extraction
- Max pooling for dimension reduction
- Fully connected layers for classification
- Dropout to prevent overfitting

### `train.py`
Trains the model and visualizes results:
- Loads preprocessed data
- Trains CNN for 10 epochs (batch size 32)
- Plots training accuracy and loss
- Saves trained model

### `predict_draw.py`
Interactive GUI application:
- Tkinter canvas for drawing
- PIL for image manipulation
- OpenCV for preprocessing (centering, resizing)
- TensorFlow for prediction

## 💡 Tips for Best Results

1. **Draw digits clearly** in the center of the canvas
2. **Use smooth strokes** - the brush is automatically sized
3. **Fill the digit well** - don't make it too thin
4. **Center your drawing** - the model expects centered digits
5. **Draw similar to handwriting** - not printed fonts

## 🐛 Troubleshooting

### Model not found error
- Make sure you run `train.py` first to create `digit_cnn.h5`

### Poor prediction accuracy
- Draw digits larger and more centered
- Make strokes thicker and clearer
- Try drawing the digit again

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (needs 3.8+)

## 📚 Learning Resources

This project demonstrates:
- **Convolutional Neural Networks (CNNs)**: Image feature extraction
- **Image Preprocessing**: Normalization, resizing, centering
- **Transfer Learning Basics**: Training and saving models
- **GUI Development**: Tkinter for interactive applications
- **Computer Vision**: Image manipulation with OpenCV and PIL

## 🎓 Next Steps

Want to extend this project? Try:
- Adding more training epochs for better accuracy
- Implementing data augmentation (rotation, scaling)
- Creating a web version with Flask or Streamlit
- Supporting multiple digits in one image
- Training on custom handwritten datasets
- Experimenting with different CNN architectures

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to improve this project! Suggestions:
- Add real-time prediction while drawing
- Include uncertainty visualization
- Support drawing with different colors
- Add save/load drawing functionality

---

**Happy Learning! 🎉**

Made with ❤️ for beginners learning Deep Learning