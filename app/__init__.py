from flask import Flask
from flask_cors import CORS
from tensorflow.keras.models import load_model
import os

# Initialize the Flask app and CORS
app = Flask(__name__)
CORS(app)

# Load the model globally
model_path = os.path.join(os.path.dirname(__file__), "model", "tumor_model.keras")
model = load_model(model_path)
class_names = ['Glioma', 'Meningioma', 'no tumor', 'Pituitary']

# Import routes to register them
from app import main
