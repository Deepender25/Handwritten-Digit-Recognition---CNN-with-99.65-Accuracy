#!/usr/bin/env python3
"""
Simple Flask web app for MNIST digit recognition.
Run this script and go to http://localhost:5000 in your browser.

Install required packages:
pip install flask pillow

Then run:
python web_app.py
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import base64
import json

app = Flask(__name__)

# Global variable to store the model
model = None

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        print("Loading model...")
        model = keras.models.load_model('mnist_digit_classifier.keras')
        print("Model loaded successfully!")
    return model

def preprocess_image_from_base64(image_data):
    """
    Preprocess image from base64 data for prediction
    
    Args:
        image_data: Base64 encoded image data
    
    Returns:
        Preprocessed image array
    """
    # Remove data URL prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64 to image
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    
    # Invert colors (drawing canvas usually has black on white,  expects white on black)
    img_array = 1.0 - img_array
    
    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

@app.route('/')
def index():
    """Main page with drawing canvas"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Digit Recognition</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                text-align: center;
            }
            #canvas {
                border: 2px solid #333;
                cursor: crosshair;
                margin: 20px 0;
            }
            button {
                padding: 10px 20px;
                margin: 10px;
                font-size: 16px;
                cursor: pointer;
            }
            #result {
                margin: 20px 0;
                padding: 20px;
                background-color: #f0f0f0;
                border-radius: 5px;
                min-height: 50px;
            }
            .confidence-bar {
                background-color: #ddd;
                border-radius: 5px;
                overflow: hidden;
                margin: 5px 0;
                height: 25px;
            }
            .confidence-fill {
                background-color: #4CAF50;
                height: 100%;
                text-align: center;
                line-height: 25px;
                color: white;
                transition: width 0.3s ease;
            }
        </style>
    </head>
    <body>
        <h1>Digit Recognition</h1>
        <p>Draw a digit (0-9) in the box below:</p>
        
        <canvas id="canvas" width="280" height="280"></canvas>
        <br>
        
        <button onclick="clearCanvas()">Clear</button>
        <button onclick="predict()">Predict</button>
        
        <div id="result">
            <p>Draw a digit and click "Predict" to see the result!</p>
        </div>
        
        <div style="text-align: left; margin-top: 30px;">
            <h3>How to use:</h3>
            <ol>
                <li>Draw a digit (0-9) in the canvas above</li>
                <li>Click "Predict" to get the AI's prediction</li>
                <li>The model will show the predicted digit and confidence</li>
                <li>Click "Clear" to draw a new digit</li>
            </ol>
            
            <h3>Tips for better accuracy:</h3>
            <ul>
                <li>Draw digits clearly and fill the canvas</li>
                <li>Make sure the digit is centered</li>
                <li>Use thick strokes for better recognition</li>
            </ul>
        </div>

        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            let isDrawing = false;
            
            // Set up canvas
            ctx.lineWidth = 15;
            ctx.lineCap = 'round';
            ctx.strokeStyle = '#000';
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Drawing functions
            function startDrawing(e) {
                isDrawing = true;
                draw(e);
            }
            
            function draw(e) {
                if (!isDrawing) return;
                
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                ctx.lineTo(x, y);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(x, y);
            }
            
            function stopDrawing() {
                if (isDrawing) {
                    ctx.beginPath();
                    isDrawing = false;
                }
            }
            
            // Event listeners
            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mouseout', stopDrawing);
            
            // Touch events for mobile
            canvas.addEventListener('touchstart', function(e) {
                e.preventDefault();
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent('mousedown', {
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });
                canvas.dispatchEvent(mouseEvent);
            });
            
            canvas.addEventListener('touchmove', function(e) {
                e.preventDefault();
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent('mousemove', {
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });
                canvas.dispatchEvent(mouseEvent);
            });
            
            canvas.addEventListener('touchend', function(e) {
                e.preventDefault();
                const mouseEvent = new MouseEvent('mouseup', {});
                canvas.dispatchEvent(mouseEvent);
            });
            
            function clearCanvas() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#fff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                document.getElementById('result').innerHTML = '<p>Draw a digit and click "Predict" to see the result!</p>';
            }
            
            function predict() {
                const imageData = canvas.toDataURL('image/png');
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({image: imageData})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                    } else {
                        let html = `
                            <h2>Prediction: ${data.prediction}</h2>
                            <p>Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
                            <h3>All Probabilities:</h3>
                        `;
                        
                        for (let i = 0; i < data.probabilities.length; i++) {
                            const prob = data.probabilities[i];
                            const percentage = (prob * 100).toFixed(1);
                            html += `
                                <div style="margin: 5px 0;">
                                    <strong>${i}:</strong>
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: ${percentage}%;">
                                            ${percentage}%
                                        </div>
                                    </div>
                                </div>
                            `;
                        }
                        
                        document.getElementById('result').innerHTML = html;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('result').innerHTML = '<p style="color: red;">Error making prediction</p>';
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from uploaded image"""
    try:
        # Load model if not already loaded
        model = load_model()
        
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess image
        img_array = preprocess_image_from_base64(image_data)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        probabilities = predictions[0].tolist()
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Digit Recognition Web App...")
    print("Make sure 'mnist_digit_classifier.keras' is in the same directory!")
    print("\nOnce started, go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
