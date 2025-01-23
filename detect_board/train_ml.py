"""
This trains the CNN that will be used to detect if an image is a chessboard. 
The model should be trained with only the data folders: chessboard, not_chessboard.
The other folders were used to manage images for training, but should to be removed.

dynamic_chessboard can save the images it thinks are chessboards/arent chessboards on the folders chessboard_try/not_chessboard_try,
use this feature to train the model and then finetune the model(or retrain it).

If a chessboard does not show up in either of them consider changing detect_chess_diagram/detect_chessboards.

Emptied the data as it had the whole woodpecker method book, to avoid copywrights
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set up image data generator
datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    'data/train',  # Point to the parent directory
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Set class mode for binary classification
)

# Print class indices
print("Class indices:", generator.class_indices)

# Build a CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(generator, epochs=25)

# Evaluate the model on the training data
train_loss, train_accuracy = model.evaluate(generator)
print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")

# Save the model for later use
model.save('chessboard_detector.h5')

# Check image paths and labels
for i in range(len(generator)):
    img, label = generator[i]  # Get the image and label
    prediction = model.predict(img)  # Get the prediction
    # Get the class label from the class indices
    class_label = list(generator.class_indices.keys())[list(generator.class_indices.values()).index(int(label[0]))]
    print(f"Image {i+1}: True Label = {label[0]} ({class_label}), Predicted Probability = {prediction[0][0]}")