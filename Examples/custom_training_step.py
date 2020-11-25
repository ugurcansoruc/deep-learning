#!/usr/bin/env python3
"""Code not working just an example"""
from tensorflow.keras import Model, Input, layers
import tensorflow as tf


class CustomModel(Model):
    def train_step(self, data):

        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


def function():
    inputs = input(shape=(32,))
    outputs = layers.Dense(1)(inputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=[...])

    model.fit(dataset, epochs=3, callbacks=...)


if __name__ == '__main__':

    function()
