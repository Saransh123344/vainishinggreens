import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array, image_dataset_from_directory
import pathlib



path_img = os.path.join('validation', 'integras')
class_names = ['Road_deforestation', 'wildfire','clear_cutting'] 
img_width = 320
img_height = 320

model = load_model('C:\\Users\\Bhavya\\Desktop\\Dataset\\deforest.keras')
#
for i in range(1, 19):
    file =  f'C:\\Users\\Bhavya\\Desktop\\Dataset\\valid\Road\\road ({i}).jpg'   
    img = tf.keras.utils.load_img(
        file, target_size=(img_width, img_height)
    )
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(file)
    print(
        "This image appears to belong to a forest {} with {:.2f}% precision."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

for i in range(1, 17):
    file = f'C:\\Users\\Bhavya\\Desktop\\Dataset\\valid\\wildfire1\\fire ({i}).jpg'    
    img = tf.keras.utils.load_img(
        file, target_size=(img_width, img_height)
    )
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(file)
    print(
        "This image appears to belong to a forest {} with {:.2f}% precision."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


    

validation_dir = pathlib.Path('C:\\Users\\Bhavya\\Desktop\\Dataset\\valid')
validation_ds = image_dataset_from_directory(
    validation_dir,
    image_size=(img_width, img_height)
) 

