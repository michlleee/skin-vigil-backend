from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

def build(input_shape=(224, 224, 3), lr=1e-3, num_classes=2,
          init='normal', activ='relu', optim='adam'):
    model = Sequential([ 
        Conv2D(64, (3, 3), padding='Same', input_shape=input_shape, activation=activ, kernel_initializer='glorot_uniform'),
        MaxPool2D((2, 2)), Dropout(0.25),
        Conv2D(64, (3, 3), padding='Same', activation=activ, kernel_initializer='glorot_uniform'),
        MaxPool2D((2, 2)), Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=init),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = RMSprop(learning_rate=lr) if optim == 'rmsprop' else Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Build & load weights
model = build()
model.load_weights('model.h5')

# ===== Predict API =====
@app.route('/DetectUpload', methods=['POST'])
def detect_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess image
        img = Image.open(file.stream).convert('RGB').resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict with model
        prediction = model.predict(img)[0]
        cls = np.argmax(prediction)
        threshold = 0.64  
        confidence = float(prediction[cls])

        if confidence >= threshold:
            result = "Cancer Detected"
        else:
            result = "No Cancer"

        return jsonify({
            'prediction': result,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== Run Flask App =====
if __name__ == '__main__':
    app.run(debug=True)
