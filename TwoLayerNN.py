import numpy as np
import matplotlib.pyplot as plt
from nn_utils import relu, relu_derivative, mse_loss, mse_loss_derivative

# initialize data
X = np.array([ -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
               0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ]).reshape(1, -1)
Y = np.array([-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134, -0.201, 
              -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396, 0.345, 0.182, 
              -0.031, -0.219, -0.321]).reshape(1, -1)

# initialize weights and biases
np.random.seed(42)
input_size = 1
hidden_size = 84  # I have experimanted with different sizes and concluded that 84 works best
output_size = 1

W1 = np.random.randn(hidden_size, input_size) * 0.1 # weights for input to hidden layer
b1 = np.random.randn(hidden_size, 1) * 0.1 # bias for hidden layer

W2 = np.random.randn(output_size, hidden_size) * 0.1 # weights for hidden to output layer
b2 = np.random.randn(output_size, 1) * 0.1 # bias for output layer
# multiply by 0.1 to scale down the value to ensure the started random weights are small
# if the weights are too large, the activations result in very large values when passed through the sigmoid function which leading to a slow learning rate

# learning rate
eta = 0.21


# forward propagation function
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1  # linear combination for hidden layer
    A1 = relu(Z1) # ReLU activation function for hidden layer
    # I have chose ReLU for this task because it fits better for a regression task then sigmoid
    Z2 = np.dot(W2, A1) + b2 # linear combination for output layer
    A2 = Z2 # Output layer
    # in the output layer, there is no activation function because its a regression task
    # adding an activation function there would restrict the output (we using ReLU, so anything less than zero becomes zero)
    return A2, A1, Z1

# backward propagation and weight update function
def backward(X, Y, A2, A1, Z1, W1, W2, b1, b2, eta):
    m = X.shape[1] # samples
    
    # output layer gradients
    dZ2 = mse_loss_derivative(A2, Y)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # hidden layer gradients
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # update weights and biases
    W1 -= eta * dW1
    b1 -= eta * db1
    W2 -= eta * dW2
    b2 -= eta * db2

    return W1, b1, W2, b2

# training the nn
def train_network(X, Y, W1, b1, W2, b2, epochs, eta):
    losses = []
    for epoch in range(epochs):
        # forward propagation
        A2, A1, Z1 = forward(X, W1, b1, W2, b2)
        
        # calculate and store loss
        loss = mse_loss(A2, Y)
        losses.append(loss)
        
        # backward propagation
        W1, b1, W2, b2 = backward(X, Y, A2, A1, Z1, W1, W2, b1, b2, eta)
    
    return W1, b1, W2, b2, losses


# function to plot training error
def plot_training_error(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Error vs Epochs')
    plt.grid(True)
    plt.show()

# function to plot actual vs predicted function
def plot_function_approximation(X, Y, W1, b1, W2, b2, epoch):
    # forward propagation with final weights and biases
    Y_pred, _, _ = forward(X, W1, b1, W2, b2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(X.flatten(), Y.flatten(), label='Actual f(x)', marker='o')
    plt.plot(X.flatten(), Y_pred.flatten(), label='NN Approximation', marker='x')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Actual vs NN Approximation after {epoch} Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

epochs = 1000
W1_final, b1_final, W2_final, b2_final, losses = train_network(X, Y, W1, b1, W2, b2, epochs, eta)
plot_training_error(losses)

# plot actual vs predicted for different epochs
for epoch_check in [10, 100, 200, 400, 1000]:
    W1_temp, b1_temp, W2_temp, b2_temp, _ = train_network(X, Y, W1, b1, W2, b2, epoch_check, eta)
    plot_function_approximation(X, Y, W1_temp, b1_temp, W2_temp, b2_temp, epoch_check)
