"""
Interactive Drawing Canvas for Digit Prediction
Draw a digit with your mouse and the CNN predicts what number it is!
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2
from tensorflow.keras.models import load_model
import os


class DigitRecognizerApp:
    """
    Interactive GUI application for handwritten digit recognition
    """
    
    def __init__(self, root):
        """
        Initialize the application
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("500x650")
        self.root.resizable(False, False)
        
        # Load the trained model
        self.model = self.load_trained_model()
        
        # Drawing canvas settings
        self.canvas_size = 400
        self.brush_size = 25
        
        # Create PIL image for drawing (white background)
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Setup GUI
        self.setup_gui()
        
        # Variables for drawing
        self.last_x = None
        self.last_y = None
    
    def load_trained_model(self):
        """
        Load the pre-trained CNN model
        
        Returns:
            model: Loaded Keras model
        """
        model_path = 'digit_cnn.h5'
        
        if not os.path.exists(model_path):
            messagebox.showerror(
                "Model Not Found",
                f"Trained model '{model_path}' not found!\n\n"
                "Please run 'train.py' first to train the model."
            )
            self.root.destroy()
            return None
        
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    
    def setup_gui(self):
        """
        Setup the graphical user interface
        """
        # Title Label
        title = tk.Label(
            self.root,
            text="Draw a Digit (0-9)",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=15
        )
        title.pack(fill=tk.X)
        
        # Canvas for drawing
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            cursor="circle"
        )
        self.canvas.pack(pady=20)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Prediction result label
        self.result_label = tk.Label(
            self.root,
            text="Prediction: ?",
            font=("Arial", 24, "bold"),
            fg="#27ae60"
        )
        self.result_label.pack(pady=10)
        
        # Confidence label
        self.confidence_label = tk.Label(
            self.root,
            text="Confidence: -",
            font=("Arial", 14),
            fg="#7f8c8d"
        )
        self.confidence_label.pack()
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=15)
        
        # Predict button
        predict_btn = tk.Button(
            button_frame,
            text="🔍 Predict",
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            width=12,
            height=2,
            command=self.predict_digit,
            cursor="hand2"
        )
        predict_btn.grid(row=0, column=0, padx=10)
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            font=("Arial", 14, "bold"),
            bg="#e74c3c",
            fg="white",
            width=12,
            height=2,
            command=self.clear_canvas,
            cursor="hand2"
        )
        clear_btn.grid(row=0, column=1, padx=10)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="Draw a single digit with your mouse, then click Predict",
            font=("Arial", 10),
            fg="#95a5a6"
        )
        instructions.pack(pady=5)
    
    def start_draw(self, event):
        """Start drawing when mouse is pressed"""
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_on_canvas(self, event):
        """Draw on canvas as mouse moves"""
        if self.last_x and self.last_y:
            # Draw on Tkinter canvas (white on black)
            self.canvas.create_line(
                self.last_x, self.last_y,
                event.x, event.y,
                fill="white",
                width=self.brush_size,
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image (black on white)
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=0,
                width=self.brush_size
            )
            
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_draw(self, event):
        """Stop drawing when mouse is released"""
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: ?")
        self.confidence_label.config(text="Confidence: -")
    
    def preprocess_image(self):
        """
        Preprocess the drawn image for model prediction
        
        Returns:
            np.array: Preprocessed image ready for CNN (1, 28, 28, 1)
        """
        # Convert PIL image to numpy array
        img = np.array(self.image)
        
        # Invert colors (drawn digit should be white on black)
        img = 255 - img
        
        # Find the bounding box of the drawn digit
        coords = cv2.findNonZero(img)
        
        if coords is None:
            # No drawing detected
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding around the digit
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Crop to bounding box
        cropped = img[y:y+h, x:x+w]
        
        # Resize to 20x20 (keeping aspect ratio)
        aspect = w / h
        if aspect > 1:
            new_w = 20
            new_h = int(20 / aspect)
        else:
            new_h = 20
            new_w = int(20 * aspect)
        
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create 28x28 image with digit centered
        final_img = np.zeros((28, 28), dtype=np.uint8)
        
        # Calculate position to center the digit
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        
        # Place resized image in center
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Normalize to [0, 1]
        final_img = final_img.astype('float32') / 255.0
        
        # Reshape for CNN: (1, 28, 28, 1)
        # 1 = batch size, 28x28 = image size, 1 = grayscale channel
        final_img = final_img.reshape(1, 28, 28, 1)
        
        return final_img
    
    def predict_digit(self):
        """
        Predict the drawn digit using the trained CNN model
        """
        if self.model is None:
            return
        
        # Preprocess the image
        processed_img = self.preprocess_image()
        
        if processed_img is None:
            messagebox.showwarning(
                "No Drawing",
                "Please draw a digit first!"
            )
            return
        
        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        
        # Get predicted digit and confidence
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit] * 100
        
        # Update UI
        self.result_label.config(text=f"Prediction: {predicted_digit}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Print all probabilities (for debugging)
        print("\nPrediction probabilities:")
        for digit in range(10):
            prob = predictions[0][digit] * 100
            bar = "█" * int(prob / 2)
            print(f"{digit}: {prob:5.2f}% {bar}")
        print(f"\nPredicted: {predicted_digit} (Confidence: {confidence:.1f}%)")


def main():
    """
    Main function to run the application
    """
    print("="*60)
    print("HANDWRITTEN DIGIT RECOGNITION - INTERACTIVE CANVAS")
    print("="*60)
    print("\nStarting application...")
    print("Instructions:")
    print("1. Draw a digit (0-9) on the black canvas")
    print("2. Click 'Predict' to see the prediction")
    print("3. Click 'Clear' to draw a new digit")
    print("\n" + "="*60 + "\n")
    
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
