from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model_vgg16 = load_model('models/model_vgg16.h5')
model_resnet = load_model('models/model_resnet.h5')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

labels = [
    '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 
    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
    'W', 'X', 'Y', 'Z', 'best of luck', 'fuck you', 'i love you', 'space'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
      
        img_array = preprocess_image(file_path)
        
        prediction_vgg16 = model_vgg16.predict(img_array)
        prediction_resnet = model_resnet.predict(img_array)
        
        predicted_label_vgg16 = labels[np.argmax(prediction_vgg16)]
        predicted_label_resnet = labels[np.argmax(prediction_resnet)]
        
        return render_template('result.html', 
                               filename=filename, 
                               prediction_vgg16=predicted_label_vgg16, 
                               prediction_resnet=predicted_label_resnet)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "_main_":
    app.run(host='0.0.0.0', port=5000,debug=True)