# Rosalia Brooks
# Partners: Alvin Gao
# brookskw
# rosalia.brooks@vanderbilt.edu
# File name: cnet.py

# Assignment 4 - EECE 4354
# April 23, 2019


"""
cnet.py

cnet.py lays out the necessary functions and classes for constructing a
convolution neural network with the PyTorch.nn library as well as other
packages in the PyTorch bundle.

Uses helper functions from other files:
Insert File Names Here
"""

from incl import mnist_loader as ml
from incl import conv_net as cn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Declare out network as outlined in conv_net
net = cn.Net()

# Get the list of learnable parameters from the net object
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight

in_im = nn.randn(1, 1, 32, 32)
out = net(in_im)
print(out)

# Zero the gradient buffers of all parameters and back propagate the net with
# random gradients.
net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(in_im)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output

# This is a Mean Square Error Loss which looks at the bias and the
# variance of a sample population and calculates the loss from it.
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

