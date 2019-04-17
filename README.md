# EECE_4354_Final_AI_Tests
This project is for the Spring 2019 EECE 4354 Computer Visions class at Vanderbilt University. The project consists of trying to implement various versions of a number recognition neural network to try and achieve a network that is robust and actually "intelligent." Kyle Brooks and Alvin Gao

To run the currently present code in console,
import mnist_loader
import network
training_data, validation_data, test_data = \
... mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

This is an example with 1000 neurons from the network.
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
