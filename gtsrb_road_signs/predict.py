import sys
import random

import cv2
import numpy as np
import pygame
import tensorflow as tf
from sklearn.model_selection import train_test_split

from traffic import load_data

label_map = {
    "0": "20_speed",
    "1": "30_speed",
    "2": "50_speed",
    "3": "60_speed",
    "4": "70_speed",
    "5": "80_speed",
    "6": "80_lifted",
    "7": "100_speed",
    "8": "120_speed",
    "9": "no_overtaking_general",
    "10": "no_overtaking_trucks",
    "11": "right_of_way_crossing",
    "12": "right_of_way_general",
    "13": "give_way",
    "14": "stop",
    "15": "no_way_general",
    "16": "no_way_trucks",
    "17": "no_way_one_way",
    "18": "attention_general",
    "19": "attention_left_turn",
    "20": "attention_right_turn",
    "21": "attention_curvy",
    "22": "attention_bumpers",
    "23": "attention_slippery",
    "24": "attention_bottleneck",
    "25": "attention_construction",
    "26": "attention_traffic_light",
    "27": "attention_pedestrian",
    "28": "attention_children",
    "29": "attention_bikes",
    "30": "attention_snowflake",
    "31": "attention_deer",
    "32": "lifted_general",
    "33": "turn_right",
    "34": "turn_left",
    "35": "turn_straight",
    "36": "turn_straight_right",
    "37": "turn_straight_left",
    "38": "turn_right_down",
    "39": "turn_left_down",
    "40": "turn_circle",
    "41": "lifted_no_overtaking_general",
    "42": "lifted_no_overtaking_trucks",
}

speed_range = {
    "FAST": 1,
    "MEDIUM": 0.5,
    "SLOW": 0.2,
}

SPEED = speed_range["MEDIUM"]

if len(sys.argv) != 2:
    sys.exit("Usage: python predict.py model")
# Load pre-trained model
model = tf.keras.models.load_model(sys.argv[1])

images, labels = load_data("gtsrb")
labels = tf.keras.utils.to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(
    np.array(images), np.array(labels), test_size=0.4
)

# Initialize Pygame
pygame.init()

# Set up display
display_width = 400
display_height = 500
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Road Image Classification (GTSRB)")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)

# Font
font = pygame.font.SysFont(None, 25)


def display_image_with_labels(img, true_label, predicted_label):
    gameDisplay.fill(white)

    img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    scaled_img_surface = pygame.transform.scale(
        img_surface, (display_width, display_height - 50)
    )
    gameDisplay.blit(scaled_img_surface, (0, 0))

    true_label_text = font.render("True Label: " + true_label, True, black)
    gameDisplay.blit(true_label_text, (10, display_height - 50))
    predicted_label_text = font.render(
        "Predicted Label: " + predicted_label, True, black
    )
    gameDisplay.blit(predicted_label_text, (10, display_height - 25))
    pygame.display.update()


def main():
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_q]:
                running = False

        # Select a random image from the test set
        idx = random.randint(0, len(x_test) - 1)
        img = x_test[idx]
        true_label = label_map[str(np.argmax(y_test[idx]))]

        # Predict label using the model
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        predicted_label = label_map[str(np.argmax(model.predict(img)))]

        # Display the image with labels
        display_image_with_labels(img[0], str(true_label), str(predicted_label))

        clock.tick(SPEED)

    pygame.quit()


if __name__ == "__main__":
    main()
