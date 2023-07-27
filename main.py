import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Model 
model = keras.Sequential(
    [
     layers.Dense(1024, activation='relu'),
     layers.Dense(512, activation='relu'),
     layers.Dense(256, activation='relu'),
     layers.Dense(10),
     
    ]
)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adagrad(learning_rate=0.005),
    metrics = ["accuracy"],

)

model.fit(x_train, y_train, batch_size = 50, epochs = 5, verbose=2)
model.fit(x_test, y_test, batch_size = 50, verbose = 2)

print(model.summary())