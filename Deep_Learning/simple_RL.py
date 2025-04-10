import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from keras import layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


x = np.linspace(-10, 2, 700)
y = x**2

x_test = np.linspace(-1, 4, 50)
y_test = x_test**2

model = keras.Sequential(
  [
   layers.Dense(1, activation='relu', use_bias=True, input_shape=(1,))
   ]
)

#Learning rate of 0.005 was chosen after hyperparameter tuning
opt = keras.optimizers.Adam(learning_rate=0.005)

model.compile(optimizer=opt,loss=tf.keras.losses.mse, metrics='accuracy') 
model.fit(x, y, epochs = 40, verbose=0)
print("Evaluate on test data")
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)
