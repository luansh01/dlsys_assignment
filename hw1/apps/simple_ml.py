"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
import logging

import sys

sys.path.append("python/")
import needle as ndl

logger = logging.getLogger(__name__)

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    #test_label_file = "../data/t10k-labels-idx1-ubyte.gz"
    #test_data_file = "../data/t10k-images-idx3-ubyte.gz"
    #train_label_file = "../data/train-labels-idx1-ubyte.gz"
    #train_data_file = "../train-images-idx3-ubyte.gz"
    with gzip.open(label_filename) as f:
        logger.debug('Start read label_file: '+ label_filename)
        magic_number = int.from_bytes(f.read(4), 'big')
        logger.debug('test_label_file magic number is ' + str(magic_number))
        if magic_number == 2049:
            num_items = int.from_bytes(f.read(4), 'big')
            logger.debug('items number is ' + str(num_items))
            labels_uint8 = np.frombuffer(f.read(), dtype = np.uint8)
            #labels = labels.reshape(num_items)
        else:
            logger.error('file format error')
            return
    with gzip.open(image_filesname) as f:
        logger.debug("Start read image_file: "+ image_filesname)
        magic_number = int.from_bytes(f.read(4), 'big')
        logger.debug('test_label_file magic number is '+ str(magic_number))
        if magic_number == 2051:
            num_items = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            logger.debug('items,row,col number is ' + str(num_items) +' '+ str(num_rows)+' '+ str(num_cols))
            images_uint8 = np.frombuffer(f.read(),dtype = np.uint8)
            images_uint8 = images_uint8.reshape(num_items,num_rows*num_cols)

            images_float32 = (images_uint8/255.0) .astype(np.float32)
        else:
            logger.error('file format error')
            return
    return  images_float32, labels_uint8
    raise NotImplementedError()
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    z_y = Z * y_one_hot
    z_y = ndl.summation(z_y, (1,))
    #print(z_y)
    z_exp = ndl.exp(Z)
    #print(z_exp)
    sum_z_exp = ndl.summation(z_exp, (1,))
    #print(sum_z_exp)
    log_sum = ndl.log(sum_z_exp)
    return ndl.summation(log_sum - z_y, (0,))/log_sum.shape[0]
    raise NotImplementedError()
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_batch = y.shape[0] // batch
    #print(W1)
    for i in range(num_batch):
        y_slice = y[i*batch:(i+1)*batch]
        X_slice = X[i*batch:(i+1)*batch]
        y_one_hot = np.zeros((y_slice.shape[0], W2.shape[1]))
        y_one_hot[np.arange(y_slice.size), y_slice] = 1

        y_tensor = ndl.Tensor(y_one_hot)
        X_tensor = ndl.Tensor(X_slice)
        #W1_tensor = ndl.Tensor(W1)
        #W2_tensor = ndl.Tensor(W2)
        loss = softmax_loss(ndl.relu(X_tensor @ W1)@W2, y_tensor)
        loss.backward()
        W1 = ndl.Tensor(W1.numpy() - lr*W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr*W2.grad.numpy())
        
    return W1, W2
    raise NotImplementedError()
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
