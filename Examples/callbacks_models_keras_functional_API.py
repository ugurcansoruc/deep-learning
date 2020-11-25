#!/usr/bin/env python3

from tensorflow.keras import Input, layers, Model, datasets, metrics, callbacks
import tensorflow as tf


def save_model_callbacks():
    return [callbacks.ModelCheckpoint(
        filepath="C:/Users/ugurc/Desktop/model_{epoch}",
        save_freq='epoch')
    ]


def tensorboard_callbacks():
    return[
        callbacks.TensorBoard(log_dir="C:/Users/ugurc/Desktop/logs")
    ]


def function():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    inputs = Input(shape=(28, 28))

    x = layers.experimental.preprocessing.Rescaling(scale=1.0/255)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=[
                  metrics.SparseCategoricalAccuracy(name="acc")])

    batch_size = 64

    train_data = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)

    val_data = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(batch_size)

    history = model.fit(
        train_data, epochs=3, validation_data=val_data, callbacks=tensorboard_callbacks())

    print(history.history)

    loss, acc = model.evaluate(val_data)
    print("loss: %.2f" % loss)
    print("acc: %.2f" % acc)

    predictions = model.predict(val_data)
    print(predictions.shape)


if __name__ == '__main__':
    function()
