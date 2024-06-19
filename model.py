import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import shutil

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from image_transformation import apply_fourier_transform, apply_wavelet_transform
import models


EPOCHS = 100
BATCH_SIZE = 128
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3  # Single channel for grayscale images
TEST_SIZE = 0.2
MAIN_STYLES_EXCLUSION = []
SUB_STYLES_EXCLUSION = ["Novelty architecture"]

#Change to False when you want to load and evaluate already trained model
TRAIN_MODEL = False


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit(f"Usage: python {sys.argv[0]} <img-db-folder> [model.weights.h5]")

    # Get image arrays, main labels and sub_labels for all image files
    images, labels, sub_labels = load_data(sys.argv[1])
    print(f"Loaded {len(set(labels))} main styles - {len(images)} images")

    # Split data into training and testing sets
    labels_encoder = OneHotEncoder(sparse_output=False)
    encoded_labels = labels_encoder.fit_transform(np.array(sub_labels).reshape(-1, 1))

    if TRAIN_MODEL:
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(encoded_labels), test_size=TEST_SIZE
        )
    else:
        x_test = np.array(images)
        y_test = np.array(encoded_labels)

    # Get a compiled neural network
    model = get_model(len(set(sub_labels)))

    if len(sys.argv) == 3:
        filename = sys.argv[2]


    if TRAIN_MODEL:
        # Generate a test dataset that appears in the folder from where you ran the script
        generate_test_dataset(x_test, y_test, labels_encoder, base_dir="test_dataset")

        # Fit model on training data
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        
        # Save model to file
        if len(sys.argv) == 3:
            model.save_weights(filename)
            print(f"Model saved to {filename}.")
    else:
        #Load model
        model.predict(np.ones((32, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
        model.load_weights(filename, skip_mismatch=False)
        # Evaluate neural network performance
        model.evaluate(x_test, y_test, verbose=2)




def load_data2(data_dir: str):
    """Used for dataset that has 25 sub-styles in the main directory"""
    images = []
    labels = []
    folder_names = os.listdir(data_dir)
    for folder_name in folder_names:
        if folder_name in SUB_STYLES_EXCLUSION:
            continue
        path = os.path.join(data_dir, folder_name)
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            labels.append(folder_name)
        print(f"{folder_name} loaded")
    return images, labels


def load_data(data_dir: str):
    """Used for dataset divided into 9 folders which have subfolders"""
    images = []
    labels = []
    sub_labels = []
    main_styles = os.listdir(data_dir)
    for main_style in main_styles:
        if main_style in MAIN_STYLES_EXCLUSION:
            continue
        main_style_path = os.path.join(data_dir, main_style)
        for sub_style in os.listdir(main_style_path):
            if sub_style in SUB_STYLES_EXCLUSION:
                continue
            sub_style_path = os.path.join(main_style_path, sub_style)
            for file in os.listdir(sub_style_path):
                img = cv2.imread(os.path.join(sub_style_path, file))
                if img is None:
                    # Ignore image if it is corrupted
                    continue
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                if IMG_CHANNELS == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


                #fourier_img = apply_fourier_transform(img)
                #wavelet_img = apply_wavelet_transform(img)

                images.append(img)
                #images.append(fourier_img)
                #images.append(wavelet_img)
                #labels.extend([main_style] * 3)
                #sub_labels.extend([sub_style] * 3)
                
                labels.append(main_style)
                sub_labels.append(sub_style)

        print(f"{main_style} loaded")
    return images, labels, sub_labels


def generate_test_dataset(x_test, y_test, labels_encoder, base_dir="test_dataset"):
    """Creates folder like structure of images that are used for testing (those that network is not using for learning).
    The structure looks like:
    -base_dir_name
        *folder_name
        *folder_name
        *...
    It generates only the leaf folders so if you use dataset with 9 folders that have subfolders there will be only the subfolders generated exactly in the base_dir
    The created images are already processed so they match the images that CNN gets, so for example they are resized to 128x128 and have 1 channel"""
    # Decode the one-hot encoded labels
    decoded_labels_test = labels_encoder.inverse_transform(y_test)
    
    # Create the "test_dataset" folder, overriding if it exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    # Create subfolders and save images
    for idx, label in enumerate(decoded_labels_test):
        label_name = label[0]  # Get the label name from the array
        label_dir = os.path.join(base_dir, label_name)
        
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # Convert the image data to a format suitable for cv2 and save it
        image_array = x_test[idx]
        image_path = os.path.join(label_dir, f"image_{idx}.png")
        cv2.imwrite(image_path, image_array)


def get_model(num_categories: int):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)`.
    The output layer should have `num_categories` units, one for each category.
    """
    return models.get_pretrained_model(num_categories)


if __name__ == "__main__":
    main()

    # To look at the images
    # images, labels, sub_labels = load_data2(sys.argv[1])
    # img = images[0]
    # print(img.shape, labels[0])
    # cv2.imshow("Original Image", images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
