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

# import to read in images
import glob
import cv2 as cv


# function to show an image, uses matplot
def imshow(img):
    img = img / 2 + 0.5  # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# class to initialize the neural network
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


'''==================== Start multiple network test ===================='''
# variables for this program
# initialization ===================================================================
num_nets = 10  # number of networks to be made
# batch_size generalizes the amount of data being passed over, groups images in 'batches'
#   training on larger batch sizes results in worse generalization but takes a lot longer to process
batch_size = 1
# training =========================================================================
every_n = 10000  # amount to print statistics by
num_epochs = 2  # number of times passing images through neural net
# evaluation =======================================================================
test_batch_size = 4  # size of example evaluation set

# initialization ==================================================================================
# create a set of neural networks and train each individually
#   at the end, determine which network evaluated to be best results
#   use that best performer to test the sample images
net = []  # a list of networks
net_score = []  # accuracy of each network
print('Starting program, make %d networks:' % num_nets)
total_train_time = 0.0
total_eval_time = 0.0

for net_id in range(num_nets):
    net.append(Net())  # initialize neural net
    if net_id == 0:
        print(net[net_id])  # all nets will have same features, so only print first one

    # MNIST database is composed of handwritten digits examples
    #   it has a training set of 60,000 examples, 50,000 examples = training and 10,000 examples = test
    # for MNIST dataset, mean and standard deviation is 0.1307 and 0.3081 respectively
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # get 60,000 training images
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    if net_id == 0:
        print('>>> total training batch number: %d' % (len(train_loader)))

    # get 10,000 test images
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    if net_id == 0:
        print('>>> total testing batch number: %d' % (len(test_loader)))

    '''
    ========================================= Start Training ==========================================
    '''
    # training ========================================================================================
    # higher learning rate (lr) will produce better accuracy, but too high may lead to false positives
    #   learning rate too low would make the network not adjust any of its weights
    optimizer = optim.SGD(net[net_id].parameters(), lr=0.001, momentum=0.9)  # using a "SGD with momentum"
    criterion = nn.CrossEntropyLoss()  # using a "Classification Cross-Entropy loss"

    training_time = 0.0  # total elapsed time during training
    if net_id == 0:
        print('>>> %d epochs, printing every %d image statistics:\n' % (num_epochs, every_n))

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
            outputs = net[net_id](inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()  # count total losses for these every_n images
            if batch_idx % every_n == (every_n - 1):  # print once every every_n mini-batches
                print('%d: ==>>> epoch: %d/%d, batch index: %5d/%d, train loss: %.6f' %
                      (net_id, epoch + 1, num_epochs, batch_idx + 1, len(train_loader), running_loss / every_n))
                running_loss = 0.0

        end_time = time.time()  # get epoch end time
        print('%d: =====>>> time taken for %d epoch: %0.6f seconds\n' % (net_id, epoch + 1, end_time - start_time))
        training_time += end_time - start_time  # add epoch time to total time
        total_train_time += training_time
    print('%d: Finished Training\n=====>>> time taken for training: %0.6f seconds' % (net_id, training_time))

    '''
    ========================================================================================= End Training
    ========================================== Start Evaluation ==========================================
    '''
    # evaluation =========================================================================================
    # testing for all evaluation images
    correct_cnt, running_loss = 0, 0
    total_cnt = 0
    eval_time = 0.0
    start_e_time = time.time()
    with torch.no_grad():  # don't back propagate this time, only forward evaluate
        for data in test_loader:
            images, labels = data
            outputs = net[net_id](images)
            _, predicted_label = torch.max(outputs.data, 1)

            total_cnt += labels.size()[0]
            correct_cnt += (predicted_label == labels).sum().item()
    end_e_time = time.time()
    eval_time += end_e_time - start_e_time
    net_score.append(100.0 * (correct_cnt / total_cnt))
    print('\n%d: Time for evaluating net: %0.6f seconds' % (net_id, eval_time))
    total_eval_time += eval_time
    print('%d: Accuracy of the neural net: %d/%d test images = %0.3f%%\n' %
          (net_id, correct_cnt, len(test_loader), net_score[net_id]))

'''
==================================================================================== End Evaluation
========================================== Start Testing ==========================================
'''
# real test =======================================================================================

# print out working time in training/evaluating
print('Networks configuration times: \nTraining: %13.6f seconds\nEvaluation: %11.6f\n' %
      (total_train_time, total_eval_time))
# determine best net[net_id] of the ones generated above
print('Network accuracy: ', ' '.join('%8.3f%%' % net_score[j] for j in range(num_nets)))
best_net_index = np.argmax(net_score)
best_net = net[best_net_index]
print('Net number ID [%d] chosen as best with %0.3f%% accuracy\n' % (best_net_index, net_score[best_net_index]))

# test with hand-drawn image set, 10 samples for each digit 0-9
images = glob.glob('./hand_digits/*.png')  # load in the images
sample_counter = 0  # keep track of proper label
for fname in images:
    # Load the image, convert to gray scale, and down-sample to 28x28
    img = cv.resize(cv.imread(fname, 0), (28, 28))  # convert gray image to down-sampled image
    img = cv.bitwise_not(img)  # invert image to match MNIST data set
    dst_dir = ('./hand_digits/%s') % str(int(sample_counter / 10))  # make directory to proper label folder
    cv.imwrite('%s/%s_%d.jpg' % (dst_dir, str(int(sample_counter / 10)), sample_counter), img)  # write to folder
    sample_counter += 1

sample_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])
sample_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(root='./hand_digits/', transform=sample_transform),
    batch_size=10, shuffle=False)
num_samples = len(sample_loader)  # number of samples loaded into array
print('Processing Total Sample Accuracy %d images...\n' % sample_counter)
samp_data_iter = iter(sample_loader)
for k in range(int(sample_counter / num_samples)):
    print('==> Testing Accuracy for [%d] samples:' % k)
    samp_images, samp_labels = samp_data_iter.next()
    imshow(torchvision.utils.make_grid(samp_images))  # print images
    samp_outputs = best_net(samp_images)
    _, samp_predicted = torch.max(samp_outputs, dim=1)

    # get confidence levels for predictions of sample images
    samp_sm = torch.nn.Softmax(samp_outputs)  # get softmax for all all samples
    samp_top_two = torch.topk(samp_outputs, 2, dim=1)  # gets the top two predictions
    samp_distribution = []  # hold final activation values for last network layer
    samp_prob1 = []  # distribution probability of the first guess
    samp_prob2 = []  # distribution probability of the second guess
    for j in range(num_samples):
        samp_distribution.append(samp_sm.dim.data[j])  # get the 0-9 distribution tensor values of one output
        samp_distribution[j].add_((-1 * torch.min(samp_distribution[j])) + 1)  # cancel all negatives, add all by smallest value
        samp_prob1.append(100.0 * (samp_labels[0].item() / torch.sum(samp_distribution[j])))  # get percentage of top guess
        samp_prob2.append(100.0 * (samp_top_two[0][j][1].item() / torch.sum(samp_distribution[j])))  # get percentage of second guess

    # print out statistics
    print('\n==>Ground Truth: ', ' '.join('%5s' % samp_labels[j].item() for j in range(num_samples)))
    print('==>Predicted:    ', ' '.join('%5s' % samp_predicted[j].item() for j in range(num_samples)))
    print('==>Confidence %: ', ' '.join('%5.2f' % samp_prob1[j].item() for j in range(num_samples)))
    print('\n==>2nd Predicted:', ' '.join('%5s' % samp_top_two[1][j][1].item() for j in range(num_samples)))
    print('==>Confidence %: ', ' '.join('%5.2f' % samp_prob2[j].item() for j in range(num_samples)))

# now find total accuracy processing the 100 sample data
sampleset_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(root='./hand_digits/', transform=sample_transform),
    batch_size=1, shuffle=False)
num_samples = len(sampleset_loader)  # number of samples loaded into array
print('\nProcessing Total Sample Accuracy %d images...\n' % num_samples)
samp_data_iter = iter(sampleset_loader)
sam_correct_cnt, sam_running_loss = 0, 0
sam_total_cnt = 0
with torch.no_grad():  # don't back propagate this time, only forward evaluate
    for data in sampleset_loader:
        images, labels = data
        outputs = best_net(images)
        _, predicted_label = torch.max(outputs.data, 1)
        sam_total_cnt += labels.size()[0]
        sam_correct_cnt += (predicted_label == labels).sum().item()

print('Accuracy of the neural net [%d]: %d/%d sample images = %0.3f%%\n' %
      (best_net_index, sam_correct_cnt, len(sampleset_loader), 100.0 * (sam_correct_cnt / sam_total_cnt)))
