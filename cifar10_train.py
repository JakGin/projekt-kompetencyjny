import tensorflow as tf
import numpy as np
from keras import datasets
from keras.utils import plot_model


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_labels = train_labels.flatten()
test_labels = test_labels.flatten()

input_shape = (32, 32, 3)

train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 3)
train_images = train_images / 255.0
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 3)
test_images = test_images / 255.0


train_labels = tf.one_hot(train_labels.astype(np.int32), depth=10)
test_labels = tf.one_hot(test_labels.astype(np.int32), depth=10)

model = tf.keras.models.load_model('cifar_model.keras')

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


model.summary()

#plot_model(model=model, show_shapes=True)
