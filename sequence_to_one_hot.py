#!/usr/bin/env python3

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np


def function():
    """
    Example:
    if text has word in vocabulary = 1 else 0
    text       = ["This is the 1st sample."] 
    vocabulary = ['[UNK]', 'the', 'sample', 'this', 'is', 'heres', 'and', '2nd', '1st']
    one-hot   =  [   0       1        1        1      1      0       0      0      1  ]
    """
    training_data = np.array([["This is the 1st sample."], [
                             "And here's the 2nd sample."]])

    vectorizer = TextVectorization(output_mode="binary", ngrams=1)

    vectorizer.adapt(training_data)

    integer_data = vectorizer(training_data)
    print(integer_data)
    print(vectorizer.get_vocabulary())


if __name__ == '__main__':
    function()
