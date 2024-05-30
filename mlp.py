# mlp.py
import numpy as np
import random
# generated w llm
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y, output):
        # Calculate the error
        error = y - output
        d_output = error * self.sigmoid_derivative(output)

        # Calculate the error for the hidden layer
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # Update the weights and biases
        self.W2 += self.a1.T.dot(d_output) * self.learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.W1 += x.T.dot(d_hidden) * self.learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, x, y, epochs=1000, printevery=100):
        avgrunlosses = []
        for i in range(epochs):
            sampled_x, sampled_y = random.choice(list(zip(x, y)))
            sampled_y = np.array([sampled_y])
            output = self.forward(sampled_x)
            loss = np.mean((sampled_y - output) ** 2)
            avgrunlosses.append(loss)
            self.backward(sampled_x, sampled_y, output)
            if i % printevery == 0 and i != 0:
                print(f"loss at {i}: {np.average(avgrunlosses):.5f}")
                avgrunlosses = []

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)

# Save this as mlp.py