from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os
from tensorflow.keras.preprocessing import image
import numpy as np


"""
Verifys if its a chessboard or not using the model trained by train_ml on this foulder
"""

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))



def get_model():
    # Load the trained model
    model = load_model(os.path.join(current_dir,  'fine_tuned_chessboard_detector.h5'), compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def clear_model_from_ram(model):
    K.clear_session()
    del model

def preprocess_image(img):
    # Desired size
    target_size = (150, 150)
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to numpy array
    img_array = image.img_to_array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize to [0, 1]
    img_array /= 255.0
    
    return img_array



def is_chessboard(model,img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)  # Get prediction
    # Since it's binary classification, threshold the prediction
    print(prediction[0][0])
    return prediction[0][0] < 0.5, prediction[0][0]  # Return a boolean and the probability



def verify(model,img):
    # img = image.load_img(img_path, target_size=(150, 150))  # Resize to match the model input
    is_board, probability = is_chessboard(model,img)
    return is_board
