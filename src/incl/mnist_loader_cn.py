# Rosalia Brooks
# Partner: Alvin Gao
# brookskw
# rosalia.brooks@vanderbilt.edu
# File name: mnist_loader_cn.py

# April 24, 2019
# 10:15 AM

"""
    Source for Code is from Michael Neilson's book "Neural Nets and Deep Learning" - Oct 2018
    http://neuralnetworksanddeeplearning.com/chap1.html
    This has been edited to properly format the data for use in a convolution neural network.
"""

"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

# Libraries
# Standard library
import _pickle as cPickle
import gzip

# Third-party libraries
import numpy as np


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    '''
    EDITS MADE
    The edits were to convert the result data into numpy 1 x 10 arrays for easy conversion
    To tensor and use in the MSELoss object.
    As well, returns each separate piece instead of sipping them together. Too many format
    conversions felt tedious and was probably very costly.
    Still trying to adjust this so that it is not converting a numpy array to a list and
    then back again. I feel that is inefficient and their must be a better way
    '''
    tr_d, va_d, te_d = load_data()
    training_inputs = np.array([np.reshape(x, (1, 1, 28, 28)) for x in tr_d[0]])
    training_results = np.array([vectorized_result(y) for y in tr_d[1]])
    validation_inputs = np.array([np.reshape(x, (1, 1, 28, 28)) for x in va_d[0]])
    validation_results = np.array([vectorized_result(y) for y in va_d[1]])
    test_inputs = np.array([np.reshape(x, (1, 1, 28, 28)) for x in te_d[0]])
    test_results = np.array([vectorized_result(y) for y in te_d[1]])
    return training_inputs, training_results, validation_inputs, validation_results, test_inputs, test_results


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
