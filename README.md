# Gesture_Refactor_and_Integration

This project demonstrates a simple Flask backend API that integrates a pre-trained gesture recognition model to classify American Sign Language (ASL) signs from uploaded images. The system uses a convolutional neural network (CNN) model trained on ASL datasets and provides an easy-to-use web interface for uploading images and displaying predictions.

## Repository Overview

The repository used for the pre-trained gesture recognition model is from Kaggle, specifically [Sign Language Recognition using VGG16 and ResNet50](https://www.kaggle.com/code/rahulmakwana/sign-language-recognition-vgg16-resnet50). This repository was chosen based on several factors:

- **Model Accuracy**: The models (VGG16 and ResNet50) are known for their high accuracy in image classification tasks, including gesture recognition.

```
VGG16:
Accuracy for test images: 99.992 %
Accuracy for evaluation images: 100.0 %

RESNET50:
Accuracy for test images: 99.95 %
Accuracy for evaluation images: 100.0 %

```
- **Documentation Quality**: The repository includes well-documented code with explanation, making it easier to understand and integrate into other projects.
- **Ease of Integration**: The code is modular and provides clear instructions for loading and using the pre-trained models, which facilitates seamless integration into a Flask backend.

## Project Structure

```bash
static
  └── uploads
templates
   ├── index.html
   └── result.html
app.py
Final_Model.h5
requirements.txt
readme.md
```


### File Descriptions

- **`app.py`**: The main Flask application script that sets up the web server, handles file uploads, processes images, and returns predictions.
- **`Final_Model.h5`**: The pre-trained gesture recognition model file.
- **`requirements.txt`**: A list of Python dependencies required to run the application.
- **`index.html`**: The homepage of the web application where users can upload images.
- **`result.html`**: The result page that displays the uploaded image and the prediction.
- **`readme.md`**: Documentation file for the project.

## Setup and Installation

Follow these steps to set up and run the project on your local machine:

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/Rani1303/Gesture_Refactor_and_Integration.git
    cd Gesture_Refactor_and_Integration
    ```

2. **Create a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the pre-trained model**

    Ensure you have the `Final_Model.h5` file in the project directory. If not, download it from the github repository and place it in the project directory.

### Running the Application

1. **Start the Flask server**

    ```bash
    python app.py
    ```

2. **Open your browser and go to**

    ```
    http://127.0.0.1:5000/
    ```

3. **Upload an image**

    - Go to the homepage, and use the upload form to select and upload an image of an ASL sign.
    - After uploading, the application will display the predicted sign on the results page.

## Code Explanation

### `app.py`

The main Flask application script handles the core functionalities:

- **Index Route (`/`)**: Renders the homepage (`index.html`).
- **Upload Route (`/upload`)**: Handles the image upload, saves the file, preprocesses it, makes a prediction using the model, and renders the result page (`result.html`).
- **Display Route (`/display/<filename>`)**: Displays the uploaded image.

### Image Preprocessing

The `preprocess_image` function converts the uploaded image to grayscale, resizes it to 64x64 pixels, and normalizes the pixel values.

```python
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((64, 64))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array
```

### Prediction and Results

The model makes a prediction, and the predicted label is displayed on the results page.

```python
prediction = model.predict(img_array)
predicted_label = labels[np.argmax(prediction)]
```

## Screenshots

### Home Page

![Screenshot 2024-05-31 125021](https://github.com/Rani1303/Gesture_Refactor_and_Integration/assets/103280525/80e3ef62-079a-4122-bdb3-e8083f507b61)

### Result Page

![Screenshot 2024-05-31 170114](https://github.com/Rani1303/Gesture_Refactor_and_Integration/assets/103280525/140fbad9-11b6-4a1e-9f96-312dd95b20f0)


## Conclusion

This project provides a simple and effective way to integrate a pre-trained gesture recognition model into a Flask web application. By following the instructions, you can set up the project locally and start recognizing ASL signs from images. The integration of the pre-trained model ensures high accuracy and ease of use, making it a practical solution for gesture recognition tasks.

## License

[MIT](LICENSE) © 2024 Rani
