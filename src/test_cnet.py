# Rosalia Brooks
# Partner: Alvin Gao
# brookskw
# rosalia.brooks@vanderbilt.edu
# File name: test_cnet.py

# April 24, 2019
# 9:47 AM

"""
test_cnet.py

test_cnet.py takes what was incrementally examined from cnet.py
and actually applies our testing data to the network. The testing
data is from the MNIST distribution, a free subsample of a much
larger sample of handwritten numbers with expected return values.
This code tests the data against our neural network, then performs
several epochs with the test data.

Ultimately, this should result in a class that returns a trained
convolution neural network that can then be passed handwritten images
to detect. This is the end goal of the project.

Batch_Size for this experiment is 1. Takes longer to train, but the loss is updated
after every image.
"""


from incl import mnist_loader_cn as ml
from incl import conv_net as cn
import torch
import torch.nn as nn
import torch.optim as optim
import time

train_sam_size = 50000
ver_sam_size = 10000
test_sam_size = 10000
num_epochs = 2
learning_rate = 0.01
max_loss = 0.5
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Convert each element of the data into forms usable by tensorflow, that is
# 1 x b x r x c floating point tensor objects. The images will be passable
# to our network and the results will be iterable by our loss function for
# our back propagation.
tdi, tdr, vdi, vdr, tti, ttr = ml.load_data_wrapper()
training_im = torch.from_numpy(tdi)
training_res = torch.from_numpy(tdr)
validation_im = torch.from_numpy(vdi)
validation_res = torch.from_numpy(vdr)
test_im = torch.from_numpy(tti)
test_res = torch.from_numpy(ttr)

# initialize our neural network and our criterion for calculating our loss function
net = cn.Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# We run our net against the training data for several epochs. For this, it computes the average
# loss for every 5000 images.
start_time = time.time()
running_loss = 0
for epoch in range(num_epochs):
    # training loop on each of our 50000 training images.
    for i in range(train_sam_size):
        optimizer.zero_grad()
        output = net(training_im[i])
        target = training_res[i].view(1, -1)
        loss = criterion(output, torch.max(target, 1)[1].long())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 5000 == 4999:    # print every 5000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5000))
            running_loss = 0.0

end_time = time.time()
print('Total Running Time [Training]: ', end_time - start_time)

# This should have trained our network. Now we can verify this training with the validation
# data

start_time = time.time()
sum_loss = 0
for j in range(ver_sam_size):
    output = net(validation_im[j])
    target = validation_res[j].view(1, -1)
    loss = criterion(output, torch.max(target, 1)[1].long())
    sum_loss += loss.item()

end_time = time.time()
print('Average Loss on 10000 Validation Images: ', sum_loss/ver_sam_size)
print('Total Running Time [Validation]: ', end_time - start_time)

# This gives us an idea of how our network is performing after training.
# Now we compare it to our test data.

start_time = time.time()
correct = 0
for i in range(test_sam_size):
    output = net(test_im[i])
    target = test_res[i].view(1, -1)
    _, guess = torch.max(target, 1)
    _, predicted = torch.max(output, 1)
    if guess == predicted:
        correct += 1
end_time = time.time()
print('Total Running Time [Test]: ', end_time - start_time)

print('Accuracy on 10000 Test Images: %d %%' % (100 * correct/test_sam_size))
