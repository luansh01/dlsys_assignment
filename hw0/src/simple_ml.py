import struct
import numpy as np
import gzip
import logging
try:
    from simple_ml_ext import *
except:
    pass


logger = logging.getLogger(__name__)

def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
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
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
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
    with gzip.open(image_filename) as f:
        logger.debug("Start read image_file: "+ image_filename)
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

    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    Z_exp = np.exp(Z)
    Z_row_sum = np.sum(Z_exp, axis=1)
    Z_log = np.log(Z_row_sum)
    indices = y
    Z_specific = Z[np.arange(Z_exp.shape[0]),indices]
    losses = Z_log - Z_specific
    average_loss = np.mean(losses)
    return average_loss
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    #print(int(X.shape[0]/batch))
    
    num_batch = int(X.shape[0]/batch)
    #print(num_batch)
    for j in range(num_batch):
        #print(j)
        num_classes = theta.shape[1] 
        logits = np.matmul(X,theta)
        exp_logits = np.exp(logits)
        sum_exp_logits = np.sum(exp_logits, axis = 1)
        grad_loss = np.zeros(theta.shape)
        for i in range(batch):
            indice_samples = i + j*batch
            unit_vector = np.zeros(num_classes)
            unit_vector[y[indice_samples]] = 1
            #print(y[i])
            z =  unit_vector - exp_logits[indice_samples] / sum_exp_logits[indice_samples]
            
            input =  X[indice_samples]
            input.reshape(-1,1)
            #print(input)
            #print(z)
            grad_loss += np.outer(input, z)
        grad_loss /=  batch
        theta += (grad_loss * lr)
            
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_batches = int(y.shape[0]/batch)
    for i in range(num_batches):
        #forward
        output_inlayer = np.matmul(X,W1)
        relu_vector = np.maximum(0,output_inlayer)
        deri_relu = np.where(relu_vector>0,1,0)
        output_relu = np.matmul(relu_vector,W2)
        #gradient of cross-entropy loss
        exp_output = np.exp(output_relu)
        sum_exp = np.sum(exp_output,axis=1)
        grad_W1 = np.zeros(W1.shape)
        grad_W2 = np.zeros(W2.shape)
        #compute gradient at a sample point each iteration
        for j in range(batch):
            index_samples = j + i*batch
            unit_vector = np.zeros(W2.shape[1])
            unit_vector[y[index_samples]] = 1
            G2 = exp_output[index_samples]/sum_exp[index_samples] - unit_vector
            grad_W2 += np.outer(relu_vector[index_samples].reshape(-1,1), G2)

            W2G2 = np.matmul(G2,W2.T)

            G1 = deri_relu[index_samples]*W2G2
            #print(G1.shape)
            #print(X[index_samples].shape)
            grad_W1 += np.outer(X[index_samples].reshape(-1,1),G1)
        grad_W1 /= batch
        grad_W2 /= batch 
        W1 -= grad_W1 * lr
        W2 -= grad_W2 * lr


    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
