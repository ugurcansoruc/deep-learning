#!/usr/bin/env python3

# Setup
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import numpy as np


# Load the data

#!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
#!tar -xf aclImdb_v1.tar.gz
#!ls aclImdb
#!ls aclImdb/test
#!ls aclImdb/train
#!cat aclImdb/train/pos/6248_7.txt
#!rm -r aclImdb/train/unsup

batch_size = 32

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "C:/Users/ugurc/Desktop/aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "C:/Users/ugurc/Desktop/aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "C:/Users/ugurc/Desktop/aclImdb/test",
)

print(
    f"Number of batches in raw_train_ds: {tf.data.experimental.cardinality(raw_train_ds)}")

print(
    f"Number of batches in raw_val_ds: {tf.data.experimental.cardinality(raw_val_ds)}")

print(
    f"Number of batches in raw_test_ds: {tf.data.experimental.cardinality(raw_test_ds)}")

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch[i])
        print(label_batch[i])

# Prepare the data
# for <br/>

import re
import string

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


max_features = 20000
embedding_dim = 128
sequence_length = 500


vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,  # max vocabulary size
    output_mode="int",
    output_sequence_length=sequence_length
)

# text-only (no labels)
text_ds = raw_train_ds.map(lambda x, y: x)
# create vocab -> call adapt
vectorize_layer.adapt(text_ds)

# Option 1:
#text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
#x = vectorize_layer(text_input)
#x = layers.Embedding(max_features + 1, embedding_dim)(x)

# Option 2:


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

# Build a model


inputs = tf.keras.Input(shape=(None,), dtype="int64")

x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(1, activation="sigmoid", name="predictons")(x)

model = tf.keras.Model(inputs, predictions)

model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.summary()


#Train the model
epochs = 3

model.fit(train_ds, validation_data=val_ds, epochs=epochs)

#Evaluate the model on the test set
model.evaluate(test_ds)

#Make an end-to-end model
#If you want to obtain a model capable of processing raw strings, you can simply create a new model (using the weights we just trained):

inputs = tf.keras.Input(shape=(1,), dtype="string")
indices = vectorize_layer(inputs)
outputs = model(indices)

end_to_end_model = tf.keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

end_to_end_model.evaluate(raw_test_ds)

