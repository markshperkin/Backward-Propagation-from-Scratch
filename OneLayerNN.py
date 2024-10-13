import numpy as np
import matplotlib.pyplot as plt
from nn_utils import sigmoid, sigmoid_derivative, mse_loss, mse_loss_derivative

# initialize data
X = np.array([[ 0.1,  0.7,  0.8,  0.8,  1.0,  0.3,  0.0, -0.3, -0.5, -1.5],
              [ 1.2,  1.8,  1.6,  0.6,  0.8,  0.5,  0.2,  0.8, -1.5, -1.3]])

Y = np.array([[1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

# initialize weights and biases
np.random.seed(42)
# generates random values from standard normal distribution
W = np.random.randn(2, 2) * 0.1 # wieghts 2 by 2 matrix
b = np.random.randn(2, 1) * 0.1 # biases 2 by one matrix
# multiply by 0.1 to scale down the value to ensure the started random weights are small
# if the weights are too large, the activations result in very large values when passed through the sigmoid function which leading to a slow learning rate

# learning rate
eta = 0.1

# forward propagation function
def forward(X, W, b):
    Z = np.dot(W, X) + b
    return sigmoid(Z), Z

# backward propagation and weight update function
def backward(X, Y, Y_pred, Z, W, b, eta):
    m = X.shape[1]  # samples

    # output layer gradients
    dZ = mse_loss_derivative(Y_pred, Y) * sigmoid_derivative(Z)
    dW = np.dot(dZ, X.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    # update weights and biases
    W -= eta * dW
    b -= eta * db

    return W, b

# training the nn
def train_network(X, Y, W, b, epochs, eta):
    losses = []
    for epoch in range(epochs):
        # forward propagation
        Y_pred, Z = forward(X, W, b)
        
        # calculate and store loss
        loss = mse_loss(Y_pred, Y)
        losses.append(loss)
        
        # backward propagation
        W, b = backward(X, Y, Y_pred, Z, W, b, eta)
    
    return W, b, losses

# function to plot training error
def plot_training_error(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Error vs Epochs')
    plt.grid(True)
    plt.show()

# function to plot decision boundary
def plot_decision_boundary(W, b, X, Y, epoch):
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z, _ = forward(grid, W, b)
    
    Z_class = np.argmax(Z, axis=0).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_class, alpha=0.6, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=np.argmax(Y, axis=0), edgecolor='k', cmap=plt.cm.Spectral)
    plt.title(f"Decision Boundary after {epoch} Epochs")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

epochs = 100
W_final, b_final, losses = train_network(X, Y, W, b, epochs, eta)
plot_training_error(losses)

# plot decision boundaries after selected epochs
for epoch_check in [3, 10, 100]:
    W_temp, b_temp, _ = train_network(X, Y, W, b, epoch_check, eta)
    plot_decision_boundary(W_temp, b_temp, X, Y, epoch_check)
