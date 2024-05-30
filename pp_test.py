# Test on mnist
from mnist import MNIST
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mndata = MNIST('./mnist')
images, labels = mndata.load_training()

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Assign the training set to x and the testing set to y
x = (x_train, x_test)
y = (y_train, y_test)


# set up network
from pp import PP, MSE, softmax
import numpy as np
import random

io = [784, 1]
constants = 100 # a, b, c, d, etc -
pp_type = "mclaurin" 

pp = PP(io, constants, pp_type=pp_type, pp_softmax=True, eta=1e-3)
print(pp)

# train
xtrain = [np.array(x_) for x_ in x[0]]
ytrain = [np.array(y_) for y_ in y[0]]
train = list(zip(xtrain, ytrain))


epochs = 1000000
avgrunlosses = []
printevery=1000
for i in range(epochs):
    sampled_x, sampled_y = random.choice(train)
    sampled_y = np.array([sampled_y])

    if i==0:
        print(sampled_y.shape)
        print(sampled_x)
        print(sampled_y)

    yhat = pp.forward(sampled_x)

    loss = MSE(sampled_y, yhat)
    avgrunlosses.append(loss)

    pp.backward(sampled_y, yhat)
    pp.update_and_zero_grad()

    if i%printevery==0 and i!=0:
        print(f"loss at {i}: {np.average(avgrunlosses):.5f}")
        avgrunlosses = []

# test


from mlp import MLP
# Set up MLP network
input_size = 784
hidden_size = 128
output_size = 10
mlp = MLP(input_size, hidden_size, output_size)

# Train MLP
xtrain = [np.array(x_) for x_ in x[0]]
ytrain = [np.eye(output_size)[y_] for y_ in y[0]]  # One-hot encoding for labels
mlp.train(xtrain, ytrain, epochs=1000, printevery=100)

# Test MLP
xtest = [np.array(x_) for x_ in x[1]]
ytest = y[1]
predictions = [mlp.predict(x_) for x_ in xtest]
accuracy = np.mean([pred == true for pred, true in zip(predictions, ytest)])
print(f"Test accuracy: {accuracy:.2f}")