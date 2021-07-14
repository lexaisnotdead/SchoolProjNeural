import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

x = np.linspace(-3, 3, 1000).reshape(-1, 1)

def f(x):
    return x * np.sin(x * 2 * np.pi) if x < 0 else -x * np.sin(x * np.pi) + np.exp(x / 2) - np.exp(0)

def baseline_model():
    model = Sequential()
    model.add(Dense(100, input_dim=1, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dense(100, input_dim=100, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dense(1, input_dim=100, activation='linear', kernel_initializer='he_normal'))

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

f = np.vectorize(f)
y = f(x)

model = baseline_model()
model.fit(x, y, epochs=300, verbose=0)

plt.scatter(x, y, color='black', antialiased=True)
plt.plot(x, model.predict(x), color='magenta', linewidth=2, antialiased=True)
plt.show()

for layer in model.layers:
    weights = layer.get_weights()
    print(weights)