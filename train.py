import matplotlib.pyplot as pyplot
import tensorflow
import numpy
import cv2
import os

def train_model(model_name, dataset):
    (training_features, training_labels), testing = dataset.load_data()

    # normalize features
    training_features = tensorflow.keras.utils.normalize(training_features, axis=1)

    # create model and add layers
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tensorflow.keras.layers.Dense(128, activation="relu")) # relu(x) = max(0, x);
    model.add(tensorflow.keras.layers.Dense(128, activation="relu"))
    model.add(tensorflow.keras.layers.Dense(10, activation="softmax")) # forces everything to add up to 1
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # train and save the model
    model.fit(training_features, training_labels, epochs=3)
    model.save(model_name + ".model")