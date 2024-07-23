import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
        linear1 = nn.Linear(dim, hidden_dim)
        norm1 = norm(hidden_dim)
        relu1 = nn.ReLU()
        dropout = nn.Dropout(drop_prob)
        linear2 = nn.Linear(hidden_dim, dim)
        norm2 = norm(dim)
        sequential1 = nn.Sequential(linear1,norm1, relu1, dropout, linear2, norm2)
        residual = nn.Residual(sequential1)
        relu2 = nn.ReLU()
        sequential2 = nn.Sequential(residual, relu2)
        return sequential2
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(dim, hidden_dim)
    relu = nn.ReLU()
    seq = [linear1, relu]
    for i in range(num_blocks):
         seq.append(ResidualBlock(dim = hidden_dim, hidden_dim=hidden_dim//2,norm=norm, drop_prob=drop_prob))
    seq.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*seq)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    sum_loss = 0.
    sum_err = 0.
    count = 0.
    if opt == None:
         model.eval()
         for i, batch in enumerate(dataloader):
            x, y = nn.Flatten()(batch[0]), batch[1]
            output = model(x)
            loss = nn.SoftmaxLoss()(output, y)
            sum_loss += loss.numpy()*y.shape[0]
            y_est = np.argmax(output.numpy(), axis=1)
            err = np.sum(y_est != y.numpy())
            sum_err += err
            count += y_est.shape[0]
    else:
         model.train()
         for i, batch in enumerate(dataloader):
            x, y = nn.Flatten()(batch[0]), batch[1]
            output = model(x)
            loss = nn.SoftmaxLoss()(output, y)
            sum_loss += loss.numpy()*y.shape[0]
            y_est = np.argmax(output.numpy(), axis=1)
            err = np.sum(y_est != y.numpy())
            sum_err += err
            count += y_est.shape[0]
            opt.reset_grad()
            loss.backward()
            opt.step()
    return sum_err/count, sum_loss/count
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        "./data/train-images-idx3-ubyte.gz", "./data/train-labels-idx1-ubyte.gz"
    )
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = ndl.data.MNISTDataset(
        "./data/t10k-images-idx3-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size
    )

    model = MLPResNet(784, hidden_dim=hidden_dim)
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
         training_err, training_loss = epoch(train_dataloader, model, optimizer)
    test_err, test_loss = epoch(test_dataloader, model)
    return training_err,training_loss,test_err,test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
