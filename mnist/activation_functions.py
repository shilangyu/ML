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


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1 - y**2


def relu(x):
    return np.maximum(x, 0)


def drelu(y):
    return (y > 0) * 1


def leaky_relu(x):
    return np.where(x > 0, x, x*0.01)


def dleaky_relu(y):
    return np.where(y > 0, 1, 0.01)
