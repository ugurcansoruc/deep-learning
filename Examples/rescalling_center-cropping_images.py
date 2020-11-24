#!/usr/bin/env python3

from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import numpy as np


def function():
    training_data = np.random.randint(
        0, 256, size=(64, 200, 200, 3)).astype("float32")

    cropper = CenterCrop(height=150, width=150)
    scaller = Rescaling(scale=1.0/255)

    output_data = scaller(cropper(training_data))

    print(f"shape: {output_data.shape}")
    print(f"min: {np.min(output_data)}")
    print(f"max: {np.max(output_data)}")


if __name__ == '__main__':
    function()
