import tensorflow as tf
import numpy as np
from keras import datasets
import model
import models

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3  # Single channel for grayscale images
CATEGORIES_NUM = 8
FILENAME = "D:\Studia\semestr6\ProjektKompetencyjny\models\\vgg16-10.weights.h5"


input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = models.get_pretrained_model(CATEGORIES_NUM)

model.predict(np.ones((32, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))

model.load_weights(FILENAME, skip_mismatch=False)

# Re-evaluate the model
#loss, acc = model.evaluate(test_images, test_labels, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

class_names = ['Baroque', 'Classical', 'Early_Christian_Medieval', 'Eclecticism', 'Modernism',
               'Neoclassicism', 'Renaissance_and_Colonialism', 'Revivalism']


model.summary()

def classify_image(img_path):
    img = tf.keras.utils.load_img(
        img_path, target_size=(IMG_WIDTH, IMG_HEIGHT)
    )
    img_array = tf.keras.utils.img_to_array(img) # 255.0
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    for i in score:
        print(i)

    return class_names[np.argmax(score)], np.max(score)