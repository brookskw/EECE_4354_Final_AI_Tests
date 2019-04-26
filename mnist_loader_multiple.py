# NAME: Alvin Gao
# VUNETID: gaoa
# EMAIL: alvin.gao@vanderbilt.edu
# Partnered with: Kyle Brookes

# referenced code from:
# https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558

# torch imports for neural net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision  # has data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc
import torchvision.transforms as transforms
import torch.optim as optim

# imports to display image plots
import matplotlib.pyplot as plt
import numpy as np

# import to track time
import time


# function to show an image, uses matplot
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # define forward function of pytorch
    # determines how channels are computed between levels
    # with pytorch, the back-propagation function is automatically made alongside the forward function
    #   will use autograd function for back propagation
    def forward(self, x):
        # Max pooling over a (2, 2) window, if the size is a square you could specify a single number
        x = F.relu(self.conv1(x))  # ReLU activation of the convolution
        x = F.max_pool2d(x, 2)  # pool to down-sample image
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))  # change tensor to linear shape
        x = F.relu(self.fc1(x))  # ReLU activation of the linear operation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# initialization ==================================================================================
# initialize neural net
net = Net()
print(net)

# MNIST database is composed of handwritten digits examples
#   it has a training set of 60,000 examples, 50,000 examples = training and 10,000 examples = test
# batch_size generalizes the amount of data being passed over, groups images in 'batches'
#   training on larger batch sizes results in worse generalization but takes a lot longer to process
batch_size = 1
# for MNIST dataset, mean and standard deviation is 0.1307 and 0.3081 respectively
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# get 60,000 training images
# all_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# all_train_loader = torch.utils.data.DataLoader(dataset=all_train_set, batch_size=batch_size, shuffle=True)
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

# # split dataset into training and validation data
# num_validation = 10000  # number of validation data
# num_training = len(all_train_loader) - num_validation  # rest is training data
# train_loader, validation_loader = torch.utils.data.random_split(all_train_loader, [num_training, num_validation])
print('>>> total training batch number: %d' % (len(train_loader)))
# print('>>> total validation batch number: %d' % (len(validation_loader)))

# get 10,000 test images
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
print('>>> total testing batch number: %d\n' % (len(test_loader)))

# training ========================================================================================
# higher learning rate (lr) will produce better accuracy, but too high may lead to false positives
#   learning rate too low would make the network not adjust any of its weights
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # using a "SGD with momentum"
criterion = nn.CrossEntropyLoss()  # using a "Classification Cross-Entropy loss"

total_time = 0.0  # total elapsed time during training
every_n = 1000  # amount to print statistics by
num_epochs = 2  # number of times passing images through neural net
print('>>> %d epochs, printing every %d image statistics:' % (num_epochs, every_n))

# epoch  = number of times running through (forward and back) the neural network
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    start_time = time.time()  # keep track of time taken
    for batch_idx, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()  # count total losses for these every_n images
        if batch_idx % every_n == (every_n - 1):  # print once every every_n mini-batches
            print('==>>> epoch: %d/%d, batch index: %5d/%d, train loss: %.6f' %
                  (epoch + 1, num_epochs, batch_idx + 1, len(train_loader), running_loss / every_n))
            running_loss = 0.0

    end_time = time.time()  # get epoch end time
    print('=====>>> time taken for %d epoch: %0.6f seconds\n' % (epoch + 1, end_time - start_time))
    total_time += end_time - start_time  # add epoch time to total time
print('Finished Training\n=====>>> time taken for training: %0.6f seconds\n' % total_time)

# testing =========================================================================================
# preliminary test, show user a visual of the test
# test batch_size images to give user an idea ot working neural net
print('Loading Example %d images...' % batch_size)
data_iter = iter(test_loader)
images, labels = data_iter.next()
imshow(torchvision.utils.make_grid(images))  # print images
outputs = net(images)
_, predicted = torch.max(outputs, dim=1)

# get confidence levels for predictions
sm = torch.nn.Softmax(outputs)  # get softmax for all batch_size outputs
top_two = torch.topk(outputs, 2, dim=1)  # gets the top two predictions
distribution = []  # hold final activation values for last network layer
prob1 = []  # distribution probability of the first guess
prob2 = []  # distribution probability of the second guess
for j in range(batch_size):
    print('Softmax values of %dth prediction: ' % sm.dim.data[j])
    distribution.append(sm.dim.data[j])  # get the 0-9 distribution tensor values of one output
    distribution[j].add_((-1 * torch.min(distribution[j])) + 1)  # cancel all negatives, add all by smallest value
    prob1.append(100.0 * (labels[0].item() / torch.sum(distribution[j])))  # get percentage of top guess
    prob2.append(100.0 * (top_two[0][j][1].item() / torch.sum(distribution[j])))  # get percentage of second guess

# print out statistics
print('Ground Truth: ', ' '.join('%5s' % labels[j].item() for j in range(batch_size)))
print('Predicted:    ', ' '.join('%5s' % predicted[j].item() for j in range(batch_size)))
print('Confidence %: ', ' '.join('%5.2f' % prob1[j].item() for j in range(batch_size)))
print()
print('2nd Predicted:', ' '.join('%5s' % top_two[1][j][1].item() for j in range(batch_size)))
print('Confidence %: ', ' '.join('%5.2f' % prob2[j].item() for j in range(batch_size)))

# now on to testing for all test images
correct_cnt, running_loss = 0, 0
total_cnt = 0
with torch.no_grad():  # don't back propagate this time, only forward evaluate
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted_label = torch.max(outputs.data, 1)

        total_cnt += labels.size()[0]
        correct_cnt += (predicted_label == labels).sum().item()

print('\nAccuracy of the neural net: %d / %d test images = %0.4f %%' %
      (correct_cnt, len(test_loader), 100.0 * (correct_cnt / total_cnt)))

