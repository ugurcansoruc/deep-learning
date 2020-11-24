#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import numpy as np


def function():
    inputs = keras.Input(shape=(None, None, 3))

    x = CenterCrop(height=150, width=150)(inputs)

    x = Rescaling(scale=1.0/255)(x)

    x = keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation="relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    num_classes = 10

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

    processed_data = model(data)
    print(processed_data.shape)

    model.summary()


if __name__ == '__main__':
    function()
