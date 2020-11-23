#!/usr/bin/env python3
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def function():
    trainin_data = np.array(
        [["This is the 1st sample"], ["And here's the 2nd sample"]])

    vectorizer = TextVectorization(output_mode="int")

    vectorizer.adapt(trainin_data)

    integer_data = vectorizer(trainin_data)

    print(integer_data)


if __name__ == '__main__':
    function()
