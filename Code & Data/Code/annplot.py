import numpy as np
import matplotlib.pyplot as plt

train_data = np.load("Data/fashion_train.npy")
test_data = np.load("Data/fashion_test.npy")

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

X_train = X_train / 255.0
X_test = X_test / 255.0


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y.astype(int)]

y_train = one_hot_encode(y_train, 5)
y_test = one_hot_encode(y_test, 5)

def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Avoid overflow
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Add epsilon to prevent log(0)
    return loss


def relu_derivative(Z):
    return Z > 0

def backward_propagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def train(X_train, y_train, X_test, y_test, hidden_size, epochs, learning_rate):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)
        
        loss = compute_loss(y_train, A2)
        
        dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, Z1, A1, Z2, A2, W2)
        
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

def experiment_hidden_sizes(X_train, y_train, X_test, y_test, hidden_sizes, epochs, learning_rate):
    accuracies = []

    for hidden_size in hidden_sizes:
        print(f"Training with hidden size: {hidden_size}")
        W1, b1, W2, b2 = train(X_train, y_train, X_test, y_test, hidden_size, epochs, learning_rate)

        y_pred = predict(X_test, W1, b1, W2, b2)

        accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
        accuracies.append(accuracy)

    return accuracies

hidden_sizes = [16, 32, 64, 128, 256, 512]

accuracies = experiment_hidden_sizes(X_train, y_train, X_test, y_test, hidden_sizes, epochs=400, learning_rate=0.01)

plt.figure(figsize=(10, 6))
plt.plot(hidden_sizes, accuracies, marker='o')
plt.title('Accuracy vs Hidden Layer Size')
plt.xlabel('Hidden Layer Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
