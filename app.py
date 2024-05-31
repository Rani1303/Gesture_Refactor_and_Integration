from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model= load_model('Final_Model.h5')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((64, 64))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

labels={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
        9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 
        17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
        25: 'Z', 26: '1', 27: '2', 28: '3', 29:'4',30:'5',31:'6',32:'7',33:'8',
        34:'9',35:'10',36:'best of luck',37:'i love you',38:'space'}

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
        
        prediction = model.predict(img_array)
        
        predicted_label = labels[np.argmax(prediction)]
    
        
        return render_template('result.html', 
                               filename=filename, 
                               prediction=predicted_label)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
