# Rosalia Brooks
# Partner: Alvin Gao
# brookskw
# rosalia.brooks@vanderbilt.edu
# File name: test_cnet.py

# April 24, 2019
# 9:47 AM


"""
vis_net.py
vis_net.py is a program used to extract visuals from the neural network in order to
have graphical representations of the process our network takes to come to a decision.
This is performed on an untrained net, saved, then performed on a trained net. This
will allow us to see the differences between random and trained neural networks.

"""

import cv2 as cv
import torch
from incl import cnet_trained as ct

orig_im = cv.imread('data/demo.bmp', 0)
oddName = ct.ConvoNet(4, 0.01)
conf1 = oddName.pass_image(orig_im, 7)
print(conf1)
oddName.train()
conf2 = oddName.pass_image(orig_im, 7)
print(str(conf2))

