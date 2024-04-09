# Installation
First download the dataset from [here](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip). Unzip it and put it inside this folder (gtsrb_road_signs)

# How to run
- To install all needed packages run `pip install -r requirements.txt`
- In order to train the model run `python traffic.py gtsrb [model.h5]`, last argument is optional
- To check how the model predicts the road signs run `python predict.py model.h5`, where model.h5 is saved model from training process


# Notes from learning the model
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 42
TEST_SIZE = 0.4
BATCH_SIZE = 32

learning 1
- Conv2D - 128, (3, 3)
- MaxPooling2D - (2, 2)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9444

learning 2 (without dropout)
- Conv2D - 128, (3, 3)
- MaxPooling2D - (2, 2)
- Dense(256)
- Dense(256)
- accuracy: 0.9263 (Worst result, overfitting perhaps)

learning 3 (with bigger dropout)
- Conv2D - 128, (3, 3)
- MaxPooling2D - (2, 2)
- Dense(256)
- Dropout(0.5)
- Dense(256)
- Dropout(0.5)
- accuracy: 0.0556 (Weirdly bad result, I guess too big dropout made a mess)
  
learning 4 (learning 1 with MaxPooling bigger)
- Conv2D - 128, (3, 3)
- MaxPooling2D - (3, 3)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9180

learning 5 (learning 1 with 1 less hidden layer)
- Conv2D - 128, (3, 3)
- MaxPooling2D - (2, 2)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.8637
  
learning 6 (learning 1 with 1 more Conv2D layer)
- Conv2D - 128, (3, 3)
- MaxPooling2D - (2, 2)
- Conv2D - 128, (3, 3)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9402


learning 7 (learning 6 with 1 more Conv2D layer)
- Conv2D - 128, (3, 3)
- Conv2D - 128, (3, 3)
- MaxPooling2D - (2, 2)
- Conv2D - 128, (3, 3)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9744 (Surprisingly good result when stacking 2 Conv2D layers one after each other)
  
learning 8 (learning 7 without 1 Conv2D layer after pooling)
- Conv2D - 128, (3, 3)
- Conv2D - 128, (3, 3)
- MaxPooling2D - (2, 2)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9674

learning 9 (learning 1 with 1 more Dense layer)
- Conv2D - 128, (3, 3)
- MaxPooling2D - (2, 2)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.0570 (Surprisingly bad result when putting too many hidden layers, what happend?)

learning 10 (learning 1 with less hidden units)
- Conv2D - 64, (3, 3)
- MaxPooling2D - (2, 2)
- Dense(128)
- Dropout(0.2)
- Dense(128)
- Dropout(0.2)
- accuracy: 0.0554 (This is very interesting and totaly unexpected, I guess there is just not enought hidden units to learn all of the features)

learning 11 (learning 7 with less units in Conv2D layers)
- Conv2D - 64, (3, 3)
- Conv2D - 64, (3, 3)
- MaxPooling2D - (2, 2)
- Conv2D - 64, (3, 3)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9717 (Satisfying result when decreased number of kernels in Conv2D layers)

learning 12 (learning 1 with more hidden units)
- Conv2D - 128, (3, 3)
- MaxPooling2D - (2, 2)
- Dense(512)
- Dropout(0.2)
- Dense(512)
- Dropout(0.2)
- accuracy: 0.9317 (More units doesn't help)


BATCH_SIZE = 128 (making batch size bigger for faster calculations)

learning 13 (learning 11 with bigger batch size)
- Conv2D - 64, (3, 3)
- Conv2D - 64, (3, 3)
- MaxPooling2D - (2, 2)
- Conv2D - 64, (3, 3)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9804 (Hmm, is this coincidence, I though bigger batch will the result smaller)

learning 14 (learning 13 with bigger batch size)
- Conv2D - 64, (3, 3)
- Conv2D - 64, (3, 3)
- MaxPooling2D - (2, 2)
- Conv2D - 64, (3, 3)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9771

learning 15 (IMG_WIDTH = 40, IMG_HEIGHT = 40)
- Conv2D - 64, (3, 3)
- Conv2D - 64, (3, 3)
- MaxPooling2D - (2, 2)
- Conv2D - 64, (3, 3)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9768

EPOCHS = 30
- Conv2D - 64, (3, 3)
- Conv2D - 64, (3, 3)
- MaxPooling2D - (2, 2)
- Conv2D - 64, (3, 3)
- Dense(256)
- Dropout(0.2)
- Dense(256)
- Dropout(0.2)
- accuracy: 0.9821