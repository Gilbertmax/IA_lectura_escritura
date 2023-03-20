#red neuronal que sea capaz de clasificar imágenes de dígitos escritos a mano (0-9)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#datos MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Datos de píxeles estén en un rango de 0 a 1
x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)
