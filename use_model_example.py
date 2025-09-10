#!/usr/bin/env python3
"""
Example script showing how to use the trained MNIST digit recognition model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

def load_trained_model(model_path="mnist_digit_classifier.keras"):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model

def preprocess_image(image_path):
    """
    Preprocess an image for prediction
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image ready for prediction
    """
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize to 28x28 pixels
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Invert colors if needed (MNIST expects white digits on black background)
    # Uncomment the next line if your image has black digits on white background
    # img_array = 1.0 - img_array
    
    # Reshape to (1, 28, 28, 1) for model input
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def predict_digit(model, image_array):
    """
    Predict the digit from preprocessed image
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
    
    Returns:
        predicted_digit, confidence_score, all_probabilities
    """
    # Make prediction
    predictions = model.predict(image_array, verbose=0)
    
    # Get the predicted digit and confidence
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return predicted_digit, confidence, predictions[0]

def visualize_prediction(image_array, predicted_digit, confidence, probabilities):
    """Visualize the image and prediction results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show the input image
    ax1.imshow(image_array.reshape(28, 28), cmap='gray')
    ax1.set_title(f'Input Image\nPredicted: {predicted_digit} (Confidence: {confidence:.3f})')
    ax1.axis('off')
    
    # Show probability distribution
    digits = range(10)
    ax2.bar(digits, probabilities)
    ax2.set_title('Prediction Probabilities')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Probability')
    ax2.set_xticks(digits)
    
    # Highlight the predicted digit
    ax2.bar(predicted_digit, probabilities[predicted_digit], color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def predict_from_array(model, digit_array):
    """
    Predict from a numpy array (useful for programmatically created digits)
    
    Args:
        model: Trained model
        digit_array: 28x28 numpy array
    """
    # Ensure correct shape and type
    if digit_array.shape != (28, 28):
        raise ValueError("Input array must be 28x28")
    
    # Normalize and reshape
    digit_array = digit_array.astype('float32') / 255.0
    digit_array = digit_array.reshape(1, 28, 28, 1)
    
    return predict_digit(model, digit_array)

def main():
    """Main function demonstrating model usage"""
    
    # Load the trained model
    model = load_trained_model()
    
    print("\n" + "="*50)
    print("MNIST Digit Recognition - Model Usage Examples")
    print("="*50)
    
    # Example 1: Using built-in MNIST test data
    print("\n1. Testing with built-in MNIST data:")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Pick a random test image
    idx = np.random.randint(0, len(x_test))
    test_image = x_test[idx]
    true_label = y_test[idx]
    
    # Preprocess and predict
    test_image_processed = test_image.reshape(1, 28, 28, 1).astype('float32') / 255.0
    predicted_digit, confidence, probabilities = predict_digit(model, test_image_processed)
    
    print(f"True label: {true_label}")
    print(f"Predicted: {predicted_digit}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Correct: {'✅' if predicted_digit == true_label else '❌'}")
    
    # Visualize
    visualize_prediction(test_image_processed, predicted_digit, confidence, probabilities)
    
    # Example 2: Create a simple digit programmatically
    print("\n2. Testing with programmatically created digit:")
    
    # Create a simple "1" digit
    simple_digit = np.zeros((28, 28))
    simple_digit[5:23, 13:15] = 255  # Vertical line
    simple_digit[22:24, 10:17] = 255  # Bottom horizontal line
    
    predicted_digit, confidence, probabilities = predict_from_array(model, simple_digit)
    print(f"Predicted: {predicted_digit}")
    print(f"Confidence: {confidence:.3f}")
    
    visualize_prediction(simple_digit.reshape(1, 28, 28, 1), predicted_digit, confidence, probabilities)
    
    # Example 3: Instructions for using your own image
    print("\n3. To use your own image:")
    print("   - Save your digit image as 'my_digit.png'")
    print("   - Uncomment the code below:")
    print()
    print("   # image_array = preprocess_image('my_digit.png')")
    print("   # predicted_digit, confidence, probabilities = predict_digit(model, image_array)")
    print("   # visualize_prediction(image_array, predicted_digit, confidence, probabilities)")

if __name__ == "__main__":
    main()
