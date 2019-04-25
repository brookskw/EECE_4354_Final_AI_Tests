# Rosalia Brooks
# Partners: Alvin Gao
# brookskw
# rosalia.brooks@vanderbilt.edu
# File name: example_net.py

# 4/24/2019
# 11:26 PM

from incl import mnist_loader
from incl import network

"""
Still to be done:
explain the parameters of each function
explain the process a little more
put sample results as commented code
explain what the sample results mean (maybe use statistics?)
'javadocs' (the documentation at the top of each file and/or class/method)
"""

# Load in data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Construct network
net = network.Network([784, 30, 10])

# Train network, show results of 30 epochs
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# This is an example with 100 neurons from the network.
net = network.Network([784, 100, 10])

# Train network, show results of 30 epochs
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
