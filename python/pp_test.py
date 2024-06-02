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
import time

io = [784, 10]
constants = 50 # a, b, c, d, etc -
pp_type = "exp" 

pp = PP(io, constants, pp_type=pp_type, pp_softmax=False, eta=1e-3)
print(pp)


# train
nums = np.zeros((10, 10))
np.fill_diagonal(nums, 1)
xtrain = [np.array(x_) for x_ in x[0]]
ytrain = [nums[:,y_] for y_ in y[0]]
train = list(zip(xtrain, ytrain))


epochs = 10_000_000
avgrunlosses = []
printevery=1000
start = time.time()
for i in range(epochs):
    sampled_x, sampled_y = random.choice(train)
    sampled_x = np.array(sampled_x)
    sampled_y = np.array(sampled_y)

    yhat = pp.forward(sampled_x)

    loss = MSE(sampled_y, yhat)
    avgrunlosses.append(loss)

    pp.backward(sampled_y, yhat)
    pp.update_and_zero_grad()

    if i%printevery==0 and i!=0:
        print(f"loss at {i}: {np.average(avgrunlosses):.5f}  t:{time.time()-start:.2f}")
        avgrunlosses = []

print(pp.forward(np.zeros(784)))
print(pp.forward(np.ones(784)))

from DrawingApp import DrawingApp
import tkinter as tk
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root, pp)  # Use the trained model 'pp'
    root.mainloop()


# # exp one
# pp_type = "exp" 

# expp = PP(io, constants, pp_type=pp_type, pp_softmax=True, eta=3e-4)
# print(expp)

# start = time.time()
# for i in range(epochs):
#     sampled_x, sampled_y = random.choice(train)

#     yhat = expp.forward(sampled_x)

#     if i == 0:
#         print(sampled_y, yhat)

#     loss = MSE(sampled_y, yhat)
#     avgrunlosses.append(loss)

#     expp.backward(sampled_y, yhat)
#     expp.update_and_zero_grad()

#     if i%printevery==0:
#         print(f"loss at {i}: {np.average(avgrunlosses):.5f}  t:{time.time()-start:.2f}")
#         avgrunlosses = []



# # test
# from mlp import MLP
# input_size = 784
# hidden_layers = [784 for _ in range(constants)]
# output_size = 2
# learning_rate = 1e-3
# epochs = 10000
# print_every = epochs // 1000

# mlp = MLP(input_size, hidden_layers, output_size, learning_rate)

# # Generate some random data for testing
# np.random.seed(42)
# X = np.random.rand(100, input_size)
# y = np.array([[1, 0] if x[0] + x[1] > 1 else [0, 1] for x in X])

# start = time.time()
# avglosses = []
# for epoch in range(epochs):
#     yhat = mlp.forward(X)
#     loss = MSE(y, yhat)
#     mlp.backward(X, y, yhat)
#     avglosses.append(loss)
    
#     if epoch % print_every == 0:
#         print(f"Epoch {epoch}, Loss: {np.average(avglosses):.4f}, Time: {time.time() - start:.2f}s")
#         avglosses = []

# print("Training complete.")