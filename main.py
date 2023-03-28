import matplotlib.pyplot as pyplot
import tensorflow
import numpy
import cv2
import os

from train import train_model

model_name = "mnist_digits"
dataset = tensorflow.keras.datasets.mnist

model = tensorflow.keras.models.load_model(model_name + ".model")