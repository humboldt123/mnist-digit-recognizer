import matplotlib.pyplot as pyplot
import tensorflow
import numpy
import cv2
import os

from train import train_model

# import the model
model_name = "mnist_digits"
dataset = tensorflow.keras.datasets.mnist
model = tensorflow.keras.models.load_model(model_name + ".model")

i = 0
while os.path.isfile(f"digits/{i}.png"):
    try:
        image = cv2.imread(f"digits/{i}.png")[:,:,0]
        image = numpy.invert(numpy.array([image]))
        prediction = model.predict(image)
        print(f"Guessing {numpy.argmax(prediction)}")
        pyplot.imshow(image[0], cmap=pyplot.cm.binary)
        pyplot.show()
    except:
        print("Error")
    finally:
        i += 1