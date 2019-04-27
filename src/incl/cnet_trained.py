# Rosalia Brooks
# Partner: Alvin Gao
# brookskw
# rosalia.brooks@vanderbilt.edu
# File name: cnet_trained.py

# April 27, 2019
# 3:43 PM

from incl import mnist_loader_cn as ml
from incl import conv_net as cn
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt  # Uncomment to include pictures


class ConvoNet(nn.Module):
    def __init__(self, epochs=2, learning_rate=0.01):
        super(ConvoNet, self).__init__()
        # This is the net, and the passed data
        self.net = cn.Net()
        self.num_epochs = epochs
        self.lr = learning_rate

        # These are values from data
        tdi, tdr, vdi, vdr, tti, ttr = ml.load_data_wrapper()
        self.training_im = torch.from_numpy(tdi)
        self.training_res = torch.from_numpy(tdr)
        self.validation_im = torch.from_numpy(vdi)
        self.validation_res = torch.from_numpy(vdr)
        self.test_im = torch.from_numpy(tti)
        self.test_res = torch.from_numpy(ttr)
        # These are values we manually change
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)
        self.training_size = 50000
        self.valid_size = 10000
        self.test_size = 10000

    # We run our net against the training data for several epochs. For this, it computes the average
    # loss for every 5000 images.
    def train(self):
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        running_loss = 0
        for epoch in range(self.num_epochs):
            # training loop on each of our 50000 training images.
            for i in range(self.training_size):
                self.optimizer.zero_grad()
                output = self.net(self.training_im[i])
                target = self.training_res[i].view(1, -1)
                loss = criterion(output, torch.max(target, 1)[1].long())
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                if i % 5000 == 4999:    # print every 5000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 5000))
                    running_loss = 0.0

        end_time = time.time()
        print('Total Running Time [Training]: ', end_time - start_time)

    # This should have trained our network. Now we can verify this training with the validation
    # data

    def verify(self):
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        sum_loss = 0
        for j in range(self.valid_size):
            output = self.net(self.validation_im[j])
            target = self.validation_res[j].view(1, -1)
            loss = criterion(output, torch.max(target, 1)[1].long())
            sum_loss += loss.item()

        end_time = time.time()
        print('Average Loss on 10000 Validation Images: ', sum_loss/self.valid_size)
        print('Total Running Time [Validation]: ', end_time - start_time)

    # This gives us an idea of how our network is performing after training.
    # Now we compare it to our test data.

    def test_self(self):
        start_time = time.time()
        correct = 0
        for i in range(self.test_size):
            output = self.net(self.test_im[i])
            target = self.test_res[i].view(1, -1)
            _, guess = torch.max(target, 1)
            _, predicted = torch.max(output, 1)
            if guess == predicted:
                correct += 1
        end_time = time.time()
        print('Total Running Time [Test]: ', end_time - start_time)

        print('Accuracy on 10000 Test Images: %d %%' % (100 * correct/self.test_size))

    def pass_image(self, im, ans):
        im = im.astype(np.float32)
        im = im / np.amax(im)

        ten_im = torch.from_numpy(np.reshape(im, (1, 1, 28, 28)))

        # This performs our first convolution, producing six 24 x 24 images that have been convolved
        # with a random filter.
        res_conv1 = F.relu(self.net.conv1(ten_im))
        cv.imwrite('data/post-t_res_conv1_' + str(self.num_epochs) + '.bmp', ConvoNet.imshow(torchvision.utils.make_grid(res_conv1, 6, None, True)))
        res_max1 = F.max_pool2d(res_conv1, (2, 2))
        cv.imwrite('data/post-t_res_max1_' + str(self.num_epochs) + '.bmp', ConvoNet.imshow(torchvision.utils.make_grid(res_max1, 6, None, True)))
        res_conv2 = F.relu(self.net.conv2(res_max1))
        cv.imwrite('data/post-t_res_conv2_' + str(self.num_epochs) + '.bmp', ConvoNet.imshow(torchvision.utils.make_grid(res_conv2, 16, None, True)))
        res_max2 = F.max_pool2d(res_conv2, 2)
        cv.imwrite('data/post-t_res_max2_' + str(self.num_epochs) + '.bmp', ConvoNet.imshow(torchvision.utils.make_grid(res_max2, 16, None, True)))

        res = res_max2.view(-1, self.net.num_flat_features(res_max2))
        res = F.relu(self.net.fc1(res))
        print('Values of Layer 1')
        print(res)
        with open('data/l1_res_t_' + str(self.num_epochs) + '.txt', 'w') as file:
            x = ConvoNet.convert_to_table(res)
            for i in range(len(x)):
                s = str(i) + ' : ' + x[i]
                file.write(s + '\n')
        res = F.relu(self.net.fc2(res))
        print('Values of Layer 2')
        print(res)
        with open('data/l2_res_t_' + str(self.num_epochs) + '.txt', 'w') as file:
            x = ConvoNet.convert_to_table(res)
            for i in range(len(x)):
                s = str(i) + ' : ' + x[i]
                file.write(s + '\n')
        res = self.net.fc3(res)
        print('Final Results')
        print(res)
        with open('data/l3_res_t_' + str(self.num_epochs) + '.txt', 'w') as file:
            x = ConvoNet.convert_to_table(res)
            for i in range(len(x)):
                s = str(i) + ' : ' + x[i]
                file.write(s + '\n')
        print('Random Guess: ', torch.max(res, 1)[1].long())
        print('Actual Number: ', ans)
        res[res < 0] = 0
        conf = res/torch.sum(res)
        return conf

    def imshow(img):
        npimg = img.detach().numpy()
        npi = npimg[0]
        for i in range(npimg.shape[0] - 1):
            npi = np.concatenate((npi, npimg[i + 1]))
    #    plt.imshow(npi, interpolation='nearest')   # Uncomment for showing the image
    #    plt.show()                                 # Uncomment for showing the image
        npi = (256*npi).astype(np.uint8)
        return npi

    def convert_to_table(rez):
        n = rez.detach().numpy()
        t2 = [str(n[0][o]) for o in range(n.shape[1])]
        return t2
