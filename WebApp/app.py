from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import keras.saving
import numpy as np
import os
import base64

app = Flask(__name__)

# Load the pre-trained model
model = keras.saving.load_model('PneumoniaPredictor.keras')

def predict_pneumonia(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    prediction = model.predict(img)
    return (1-prediction)*100, prediction*100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', message='No file part')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('predict.html', message='No selected file')

        if file:
            # Create 'uploads' folder if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            
            img_path = os.path.join('uploads', file.filename)
            file.save(img_path)
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            normal_percent, pneumonia_percent = predict_pneumonia(img_path)
            normal_percent_list = normal_percent.tolist()
            pneumonia_percent_list = pneumonia_percent.tolist()
            prediction_results = {
                'normalPercent': normal_percent_list,
                'pneumoniaPercent': pneumonia_percent_list
            }
            return jsonify(prediction_results)
    else:
        # Handle GET request for the predict page
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
