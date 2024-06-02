# WAIT
# EACH TERM CANNOT APPROXIMATE ARBITRARY FUNCTIONS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# SHIIIIIII

# SOLUTION
# REPLACE EACH TERM W AN MLP
# BACKPROP IS THE SAME FOR EACH MLP!!!! AAAAAAAAAAA LETS GOOOOOOOOOOOOOOOOOOOOOOOOOOO
# DUDE EXPONENTIAL MULTI LAYER PERCEPTRONS!!!




# Test on mnist
from mnist import MNIST
from sklearn.model_selection import train_test_split
from DrawingApp import DrawingApp
from pp import PP, MSE, softmax
import numpy as np
import random
import time
import tkinter as tk
from PIL import Image, ImageDraw

# Load the MNIST dataset
mndata = MNIST('./mnist')
images, labels = mndata.load_training()

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Assign the training set to x and the testing set to y
x = (x_train, x_test)
y = (y_train, y_test)

# Set up network
io = [784, 10]
constants = 3  # a, b, c, d, etc -
pp_type = "mclaurin"
pp = PP(io, constants, pp_type=pp_type, pp_softmax=True, eta=1e-6)
print(pp)

# Train
nums = np.zeros((10, 10))
np.fill_diagonal(nums, 1)
xtrain = [np.array(x_) for x_ in x[0]]
ytrain = [nums[:, y_] for y_ in y[0]]
train = list(zip(xtrain, ytrain))
epochs = 100000
dtype = 'float'
avgrunlosses = []
printevery = 100
start = time.time()

for i in range(epochs):
    sampled_x, sampled_y = random.choice(train)
    yhat = pp.forward(sampled_x)
    loss = MSE(sampled_y, yhat)
    avgrunlosses.append(loss)
    pp.backward(sampled_y, yhat)
    pp.update_and_zero_grad()
    if i % printevery == 0 and i != 0:
        print(f"loss at {i}: {np.average(avgrunlosses):.5f} t:{time.time()-start:.2f}")
        avgrunlosses = []

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
#     if i % printevery == 0:
#         print(f"loss at {i}: {np.average(avgrunlosses):.5f} t:{time.time()-start:.2f}")
#         avgrunlosses = []

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root, pp)  # Use the trained model 'pp'
    root.mainloop()