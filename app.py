from __future__ import division, print_function
import sys
import os
# Disable oneDNN custom operations and suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf

# Define the class labels for your custom model
class_labels = ['EarlyPreB', 'PreB', 'ProB', 'Benign']  # Updated labels

# Define a Flask app
app = Flask(__name__)

# Define model paths
MODEL_PATHS = {
    'EfficientNetB0': 'models/EfficientNetB0.tflite',
    'MobileNetV2': 'models/MobileNetV2.tflite',
    'NasNetMobile': 'models/NasNetMobile.tflite'
}

# Function to load the TFLite model and allocate tensors
def load_model(model_name):
    model_path = MODEL_PATHS.get(model_name)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess the image and predict the class using the TFLite model
def model_predict(img_path, interpreter):
    try:
        # Load the image with the target size that matches the input size of the model (224x224)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

        # Convert the image to an array format suitable for the model
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        # Preprocess the image for MobileNetV2
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # Adjust based on the model

        # Set the tensor to the input of the model
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        preds = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

        # Ensure predictions match the number of classes
        if preds.shape[1] != len(class_labels):
            raise ValueError(f"Model output shape {preds.shape[1]} doesn't match the expected number of classes ({len(class_labels)}).")

        return preds
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

# Main route to render the homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Get the selected model from the form
            selected_model = request.form['model']
            interpreter = load_model(selected_model)

            # Get the uploaded file from the request
            f = request.files['file']

            # Save the file to the uploads directory
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make a prediction using the uploaded image
            preds = model_predict(file_path, interpreter)

            if preds is None:
                return "Prediction failed."

            # Get the index of the class with the highest probability
            pred_class_idx = np.argmax(preds, axis=1)[0]

            # Ensure the predicted index is within the range of available class labels
            if pred_class_idx < len(class_labels):
                result = class_labels[pred_class_idx]
            else:
                result = "Prediction index out of range."

            # Return the predicted class as the result
            return render_template('result.html', result=result, image_path=f.filename)
        except Exception as e:
            print(f"Error during upload or prediction: {str(e)}")
            return "Error during prediction."
    return None


@app.route('/index', methods=['GET'])
def home_redirect():
    return redirect(url_for('index'))

# Route to render the help page
@app.route('/help', methods=['GET'])
def help():
    return render_template('help.html')

# Route to render the login page
@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')  # Create this login.html file

# Route for logout action
@app.route('/logout', methods=['GET'])
def logout():
    # Implement logout functionality, e.g., clear session data
    return redirect(url_for('index'))

# Route for About Leukemia page
@app.route('/about_leukemia', methods=['GET'])
def about_leukemia():
    return render_template('about_leukemia.html')

# Route for Precautions page
@app.route('/precautions', methods=['GET'])
def precautions():
    return render_template('precautions.html')

# Route for Basic Treatment page
@app.route('/basic_treatment', methods=['GET'])
def basic_treatment():
    return render_template('basic_treatment.html')

# Route to handle image uploads and predictions

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
