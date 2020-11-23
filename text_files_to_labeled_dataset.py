#!/usr/bin/env python3

import tensorflow as tf
print(tf.__version__)


def function():
    """
    main_directory/
    ...class_a/
    ......a_text_1.txt
    ......a_text_2.txt
    ...class_b/
    ......b_text_1.txt
    ......b_text_2.txt
    """
    dataset = tf.keras.preprocessing.text_dataset_from_directory(
        'C:/Users/ugurc/Desktop/main_directory',
        batch_size=64
    )

    for data, labels in dataset:
        print(data.shape)
        print(data.dtype)
        print(labels.shape)
        print(labels.dtype)


if __name__ == '__main__':
    function()
