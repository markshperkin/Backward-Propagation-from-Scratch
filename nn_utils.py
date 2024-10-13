import numpy as np

# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid activation funcion
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# mean square error loos function
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# derivative of mean square error function
def mse_loss_derivative(y_pred, y_true):
    return (y_pred - y_true)

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# derivative of ReLU activation function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

