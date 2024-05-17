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
STYLES_EXCLUSION = ["Novelty architecture"]


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit(f"Usage: python {sys.argv[0]} <img-db-folder> [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    print(f"Loaded {len(images)} images")

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


def load_data(data_dir: str):
    images = []
    labels = []
    folder_names = os.listdir(data_dir)
    for folder_name in folder_names:
        if folder_name in STYLES_EXCLUSION:
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


def get_model(num_categories: int):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 1)`.
    The output layer should have `num_categories` units, one for each category.
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, (2, 2), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
            ),
            tf.keras.layers.Conv2D(
                64, (2, 2), activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(
                64, (2, 2), activation="relu"
            ),
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

    # For look at images
    # images, labels = load_data(sys.argv[1])
    # img = images[0]
    # print(img.shape, labels[0])
    # cv2.imshow("Original Image", images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
