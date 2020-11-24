#!/usr/bin/env python3

from tensorflow.keras import layers, datasets, Input, Model, metrics
import tensorflow as tf


def function():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    inputs = Input(shape=(28, 28))

    x = layers.experimental.preprocessing.Rescaling(scale=1.0/255)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[
                  metrics.SparseCategoricalAccuracy(name="acc")])

    batch_size = 64
    print("Fit on NumPy data")

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

    print(history.history)

    # dataset object include x_train, y_train_ batch_size

    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)
    print("Fit on Dataset")
    history = model.fit(dataset, epochs=1)

    print(history.history)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(batch_size)

    history = model.fit(dataset, epochs=1, validation_data=val_dataset)

    print("Fit with validation data")
    print(history.history)


if __name__ == '__main__':
    function()
