import tensorflow as tf
import numpy as np
import pywt

IMG_WIDTH = 128
IMG_HEIGHT = 128

def apply_fourier_transform(img):
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return magnitude_spectrum

def apply_wavelet_transform(img):
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))

def generate_imgages(path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
    )

    # Training data generator
    train_generator = datagen.flow_from_directory(
        'path_to_dataset_directory',  # Update with the correct path
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # Validation data generator
    validation_generator = datagen.flow_from_directory(
        'path_to_dataset_directory',  # Update with the correct path
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )