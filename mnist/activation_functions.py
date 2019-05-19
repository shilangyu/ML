import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1-y)


def softmax(x):
    x -= np.max(x)
    x = np.exp(x - np.max(x))
    return x / x.sum()


def dsoftmax(y):
    return y * (-y + 1)
