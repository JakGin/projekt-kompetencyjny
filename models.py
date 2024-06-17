import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

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


def get_model_Adam_512(num_categories: int):
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
        
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_categories, activation="softmax"),
        ]
    )

    # Compile the model with an optimizer and learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.1**(epoch / 10))

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, lr_schedule


def get_pretrained_model(num_categories: int):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    base_model.trainable = False  # Freeze the base model

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_categories, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model