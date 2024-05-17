import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


EPOCHS = 10
BATCH_SIZE = 128
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1  # Single channel for grayscale images
TEST_SIZE = 0.2
MAIN_STYLES_EXCLUSION = []
SUB_STYLES_EXCLUSION = ["Novelty architecture"]


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit(f"Usage: python {sys.argv[0]} <img-db-folder> [model.h5]")

    # Get image arrays, main labels and sub_labels for all image files
    images, labels, sub_labels = load_data(sys.argv[1])
    print(f"Loaded {len(set(labels))} main styles - {len(images)} images")

    # Split data into training and testing sets
    labels_encoder = OneHotEncoder(sparse_output=False)
    encoded_labels = labels_encoder.fit_transform(np.array(labels).reshape(-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(encoded_labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model(len(set(labels)))

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img)
                labels.append(main_style)
                sub_labels.append(sub_style)
        print(f"{main_style} loaded")
    return images, labels, sub_labels


def get_model(num_categories: int):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)`.
    The output layer should have `num_categories` units, one for each category.
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64,
                (2, 2),
                activation="relu",
                input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
            ),
            tf.keras.layers.Conv2D(64, (2, 2), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (2, 2), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_categories, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()

    # To look at the images
    # images, labels, sub_labels = load_data2(sys.argv[1])
    # img = images[0]
    # print(img.shape, labels[0])
    # cv2.imshow("Original Image", images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
