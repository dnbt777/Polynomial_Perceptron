import numpy as np
import random
import time

def MSE(y, yhat):
    return np.mean((y - yhat)**2)

def MSE_derivative(y, yhat):
    return 2 * (yhat - y) / y.size

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=1e-4):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]
        
    def forward(self, x):
        self.activations = [x]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = relu(z)
            self.activations.append(a)
        
        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = softmax(z)
        self.activations.append(a)
        
        return a
    
    def backward(self, x, y, yhat):
        m = y.shape[0]
        
        # Output layer gradients
        dz = yhat - y
        dW = np.dot(self.activations[-2].T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        
        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i+1].T) * relu_derivative(self.z_values[i])
            dW = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

def test_mlp():
    input_size = 2
    hidden_layers = [4, 100, 2, 3]
    output_size = 2
    learning_rate = 1e-3
    epochs = 10000
    print_every = epochs // 10
    
    mlp = MLP(input_size, hidden_layers, output_size, learning_rate)
    
    # Generate some random data for testing
    np.random.seed(42)
    X = np.random.rand(100, input_size)
    y = np.array([[1, 0] if x[0] + x[1] > 1 else [0, 1] for x in X])
    
    start = time.time()
    for epoch in range(epochs):
        yhat = mlp.forward(X)
        loss = MSE(y, yhat)
        mlp.backward(X, y, yhat)
        
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Time: {time.time() - start:.2f}s")
    
    print("Training complete.")

if __name__ == "__main__":
    test_mlp()