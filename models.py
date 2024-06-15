import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

def get_model_Adam_128(num_categories: int):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64,
                (2, 2),
                activation="relu",
                input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
        
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_categories, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 * 0.9**(epoch / 10))

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model, lr_schedule


def get_model_Adam_256(num_categories: int):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
        
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_categories, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 * 0.9**(epoch / 10))

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model, lr_schedule