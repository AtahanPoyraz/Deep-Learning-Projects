import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from keras.api.datasets import mnist
from keras.api.models import Sequential, save_model
from keras.api.layers import Dense, MaxPooling2D, Conv2D, Flatten, Input, Dropout, RandomFlip, RandomZoom, BatchNormalization
from keras.api.utils import to_categorical

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class Model():
    def load_dataset(self):
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42, test_size=0.2)

        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_valid = X_valid.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        y_train = to_categorical(y_train, 10)
        y_valid = to_categorical(y_valid, 10)
        y_test = to_categorical(y_test, 10)

        return (X_train, X_valid, X_test), (y_train, y_valid, y_test)
    
    def build(self):
        (X_train, X_valid, X_test), (y_train, y_valid, y_test) = self.load_dataset()

        self.model = Sequential([
            Input(shape=(28, 28, 1)),
            RandomFlip("horizontal_and_vertical", seed=42),
            RandomZoom(0.15, 0.15, fill_mode="reflect"),

            Conv2D(28, kernel_size=(3,3), activation="relu"),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

            Conv2D(54, kernel_size=(3,3), activation="relu"),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

            Conv2D(108, kernel_size=(3,3), activation="relu"),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

            Flatten(),
            BatchNormalization(),
            Dense(216, activation="relu"),
            Dense(10, activation="softmax"),
        ])

        print(self.model.summary())

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        history = self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=256)
        score = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"SCORE: {score}")
        
        pd.DataFrame(history.history).plot(figsize=(8,5), grid=True)
        plt.savefig("training_history.png")
        
        save = input("Press ENTER to Save The Model.")

        if save == "":
            save_model(self.model, "new_model.keras", overwrite=True)

if __name__ == "__main__":
    model_instance = Model() 
    model_instance.build()

