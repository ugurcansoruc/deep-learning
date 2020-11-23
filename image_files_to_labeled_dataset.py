#!/usr/bin/env python3

import tensorflow as tf
print(tf.__version__)


def function():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/ugurc/Desktop/main_directory',
        batch_size=64,
        image_size=(200, 200)
    )

    for data, labels in dataset:
        print(data.shape)
        print(data.dtype)
        print(labels.shape)
        print(labels.dtype)


if __name__ == '__main__':
    function()
