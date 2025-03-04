import os
import cv2
import numpy as np
import pandas as pd
import random as rd
from matplotlib import pyplot as plt

from keras.api.models import Sequential, save_model
from keras.api.layers import Dense, MaxPooling2D, Conv2D, Flatten, Input, Dropout, RandomFlip, RandomZoom, BatchNormalization

from sklearn.model_selection import train_test_split

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# DATASET : https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification

class Model():
    def __init__(self, path):
        self.path = path

    def load_dataset(self, path, dsize):
        images = []
        labels = []

        files = os.listdir(path=path)
        rd.shuffle(files)

        for file in files:
            if "cat" in file:
                image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
                image = cv2.resize(image, dsize=(dsize,dsize))
                images.append(image)
                labels.append([1])

            elif "dog" in file:
                image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
                image = cv2.resize(image, dsize=(dsize,dsize))
                images.append(image)
                labels.append([0])

        X = np.array(images).reshape(-1, dsize, dsize, 3) / 255.0
        y = np.array(labels)

        return X, y
    
    def build(self) -> None:
        X, y = self.load_dataset(self.path, dsize=64)
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42, test_size=0.2)

        self.model = Sequential([
            Input(shape=(64, 64, 3)),
            RandomFlip("horizontal_and_vertical", seed=42),
            RandomZoom(0.15, 0.15, fill_mode="reflect"),

            Conv2D(32, kernel_size=(3,3), activation="relu"),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

            Conv2D(64, kernel_size=(3,3), activation="relu"),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

            Conv2D(128, kernel_size=(3,3), activation="relu"),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

            Conv2D(256, kernel_size=(3,3), activation="relu"),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

            Flatten(),
            BatchNormalization(),
            Dense(512, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])

        print(self.model.summary())

        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        history = self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=64)
        score = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"SCORE: {score}")
        
        pd.DataFrame(history.history).plot(figsize=(8,5), grid=True)
        plt.show()
        
        save = input("Press ENTER to Save The Model.")

        if save == "":
            save_model(self.model, "./artifacts/new_model.keras", overwrite=True)

if __name__ == "__main__":
    model_instance = Model("./data/cat_dog") 
    model_instance.build()