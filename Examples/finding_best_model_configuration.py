#!/usr/bin/env python3

import tensorflow as tf
from kerastuner import RandomSearch, Hyperband


def build_model(hp):
    inputs = tf.keras.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(units=hp.Int(
        'units', min_value=32, max_value=512, step=32), activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice(
        'learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def function():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    batch_size = 64

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(batch_size)

    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='C:/Users/ugurc/Desktop/logs',
        project_name='first_tuner')

    tuner.search(train_dataset,
                 epochs=1,
                 validation_data=val_dataset)
    best_hps = tuner.get_best_hyperparameters(num_trials =1)[0]
    print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")


if __name__ == '__main__':
    function()
