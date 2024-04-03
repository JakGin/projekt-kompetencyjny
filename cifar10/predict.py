import sys
import tensorflow as tf
import pygame
import numpy as np
import random

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

# Load CIFAR-10 dataset
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Initialize Pygame
pygame.init()

# Set up display
display_width = 300
display_height = 350
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("CIFAR-10 Image Classification")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)

# Font
font = pygame.font.SysFont(None, 25)


def display_image_with_labels(img, true_label, predicted_label):
    gameDisplay.fill(white)
    
    img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    scaled_img_surface = pygame.transform.scale(img_surface, (display_width, display_height - 50))
    gameDisplay.blit(scaled_img_surface, (0, 0))

    true_label_text = font.render("True Label: " + class_names[true_label], True, black)
    gameDisplay.blit(true_label_text, (10, display_height - 50))
    predicted_label_text = font.render(
        "Predicted Label: " + class_names[predicted_label], True, black
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
        true_label = y_test[idx][0]

        # Predict label using the model
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        predicted_label = np.argmax(model.predict(img))

        # Display the image with labels
        display_image_with_labels(img[0], true_label, predicted_label)

        clock.tick(SPEED)

    pygame.quit()


if __name__ == "__main__":
    main()
