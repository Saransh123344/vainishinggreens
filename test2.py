import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array, image_dataset_from_directory
import pathlib

path_img = os.path.join('validation', 'integras')
class_names = ['Road_deforestation', 'wildfire'] 
img_width = 320
img_height = 320

model = load_model('C:\\Users\\Bhavya\\Desktop\\Dataset\\deforest.keras')

# Open the text file for writing in append mode
with open("predictions.txt", "a") as f:

  # Loop through road images
  for i in range(1, 19):
    file = os.path.join('C:\\Users\\Bhavya\\Desktop\\Dataset\\valid\\Road',f'road ({i}).jpg')
    img = tf.keras.utils.load_img(
        file, target_size=(img_width, img_height)
    )
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Write prediction results to the file
    f.write(f"{file}\n")
    f.write(
        "This image appears to belong to a forest {} with {:.2f}% precision.\n"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

  # Loop through wildfire images
  for i in range(1, 17):
    file = os.path.join('C:\\Users\\Bhavya\\Desktop\\Dataset\\valid\\wildfire1',f'fire ({i}).jpg')
    img = tf.keras.utils.load_img(
        file, target_size=(img_width, img_height)
    )
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Write prediction results to the file
    f.write(f"{file}\n")
    f.write(
        "This image appears to belong to a forest {} with {:.2f}% precision.\n"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

# Rest of the code for validation_ds (not modified)

validation_dir = pathlib.Path('C:\\Users\\Bhavya\\Desktop\\Dataset\\valid')
validation_ds = image_dataset_from_directory(
    validation_dir,
    image_size=(img_width, img_height)
)
