import os
import numpy as np
from flask import Flask, render_template, request, jsonify
# CRITICAL IMPORT FIX: Include Image and ImageEnhance from PIL
from PIL import Image, ImageEnhance
from tensorflow.keras.models import load_model
# Ensure load_img and img_to_array are imported for preprocessing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Optimizer import is no longer needed in this simple load format

# --- CONFIGURATION ---

app = Flask(__name__)

# Get the base directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Directory for temporary file uploads
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model path construction using absolute path
MODEL_NAME = 'BrainTumourDetection.h5'
MODEL_PATH = os.path.join(BASE_DIR, 'models', MODEL_NAME)

# Model parameters consistent with training
IMAGE_SIZE = 128 
UNIQUE_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary'] 

# --- MODEL LOADING (CLEANED) ---

model = None
try:
    # Use simple load_model, assuming the newly downloaded model is clean
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None


# --- PREPROCESSING FUNCTIONS (CRITICAL FIX) ---

def augment_image_for_pred(img):
    """
    Applies the same enhancements (brightness, contrast, sharpness) used 
    during training data generation.
    """
    # Apply enhancements to the PIL Image object
    img = ImageEnhance.Brightness(img).enhance(2)
    img = ImageEnhance.Contrast(img).enhance(2)
    img = ImageEnhance.Sharpness(img).enhance(2)
    
    # Convert the enhanced PIL image to a NumPy array and normalize (0-1)
    image_array = np.array(img) / 255.0
    return image_array


def preprocess_image(image_path):
    """Loads, resizes, enhances, and normalizes the image for model prediction."""
    # 1. Load the image as a PIL image, resizing it to 128x128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    
    # 2. Apply enhancements and normalization (CRITICAL STEP)
    img_array = augment_image_for_pred(img)
    
    # 3. Add batch dimension (1, 128, 128, 3) for the model input
    img_input = np.expand_dims(img_array, axis=0)
    
    return img_input


# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Internal server error: Model failed to load.'}), 500

    if 'imagefile' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['imagefile']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        # 1. Preprocess image
        img_input = preprocess_image(filepath)
        
        # 2. Predict
        predictions = model.predict(img_input)
        
        # 3. Get results
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class_index]
        predicted_label = UNIQUE_LABELS[predicted_class_index]
        
        # Return results as JSON
        return jsonify({
            'prediction': predicted_label,
            'confidence': f"{confidence * 100:.2f}%",
            'is_tumor': predicted_label != 'notumor'
        })
        
    except Exception as e:
        print(f"Error during prediction in /predict route: {e}")
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# --- RUN SERVER ---

if __name__ == '__main__':
    app.run(debug=True)