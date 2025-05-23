from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from PIL import Image
import numpy as np
import os

# Force TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)
CORS(app, origins=["https://skin-vigil-frontend.vercel.app/"])

model = None  # Will be initialized later

def build(input_shape=(224, 224, 3), lr=1e-3, num_classes=2,
          init='normal', activ='relu', optim='adam'):
    model = Sequential([
        tf.keras.Input(shape=input_shape),  # ✅ Preferred over input_shape in Conv2D
        Conv2D(64, (3, 3), padding='same', activation=activ, kernel_initializer='glorot_uniform'),
        MaxPool2D((2, 2)), Dropout(0.25),
        Conv2D(64, (3, 3), padding='same', activation=activ, kernel_initializer='glorot_uniform'),
        MaxPool2D((2, 2)), Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=init),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = RMSprop(learning_rate=lr) if optim == 'rmsprop' else Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Load model only once when the app receives its first request
@app.before_first_request
def load_model():
    global model
    model = build()
    model.load_weights('model.h5')

@app.route("/")
def home():
    return "SkinVigil backend is running."

@app.route('/api/predict', methods=['POST'])
def detect_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB').resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0]
        cls = np.argmax(prediction)
        confidence = float(prediction[cls])
        threshold = 0.64

        result = "Cancer Detected" if confidence >= threshold else "No Cancer"

        return jsonify({
            'prediction': result,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
