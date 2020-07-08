from math import ceil
from typing import List

import numpy as np

import activation_functions
from NN import NN


class ConvLayerConfig:
    def __init__(self, amount_of_filters, filter_size, pool_size, pool_func):
        assert filter_size % 2 == 1, "Only odd sized filters are supported"

        self.amount_of_filters = amount_of_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.pool_func = pool_func


class CNN:
    def __init__(self, layers: List[ConvLayerConfig], input_channels, actifunc, nn: NN):
        self.actifunc = actifunc
        self.nn = nn

        prev_depth = input_channels
        self.filters = []
        self.poolings = []
        for layer in layers:
            self.filters.append(np.random.rand(
                layer.amount_of_filters, prev_depth, layer.filter_size, layer.filter_size) * 2 - 1)
            self.poolings.append(
                {'pool_size': layer.pool_size, 'pool_func': layer.pool_func})
            prev_depth = layer.amount_of_filters

    def feedforward_convolutions(self, input):
        nodesave = [input]

        for filters, pooling in zip(self.filters, self.poolings):
            nodesave.append(self.actifunc(
                self.convolve(nodesave[-1], filters)))
            nodesave.append(
                self.pool(nodesave[-1], pooling['pool_size'], pooling['pool_func']))
        return nodesave[1:]

    def feedforward(self, input):
        nodesave = self.feedforward_convolutions(input)

        self.nn.feedforward(nodesave[-1].ravel())

        nodesave.extend(self.nn.nodesave)

        return nodesave

    def backpropagate(self, input, goal_output):
        self.nn.backpropagate(self.feedforward_convolutions(
            input)[-1].ravel(), goal_output)

    def predict(self, input):
        return np.argmax(self.feedforward(input)[-1])

    @staticmethod
    def convolve(matrix, filters):
        f_amt, f_depth, f_size, _ = filters.shape
        m_depth, m_height, m_width = matrix.shape
        assert f_depth == m_depth, "Filter's depth is different than input matrix depth"

        out = np.zeros(
            (f_amt, m_height - (f_size // 2)*2, m_width - (f_size // 2)*2))

        for n, filter in enumerate(filters):
            flatten_filter = filter.ravel()
            for i in range(1, out.shape[1] + 1):
                for j in range(1, out.shape[2] + 1):
                    out[n][i-1][j-1] = matrix[:, i-1:i+f_size-1, j -
                                              1:j+f_size-1].ravel().dot(flatten_filter)

        return out

    @staticmethod
    def pool(matrix, size, func):
        m_depth, m_height, m_width = matrix.shape

        # pad if needed
        pad_height = size - m_height % size if m_height % size != 0 else 0
        pad_width = size - m_width % size if m_width % size != 0 else 0
        padded = np.pad(matrix, ((0, 0), (0, pad_height), (0, pad_width)),
                        'constant', constant_values=0)

        out = np.zeros((m_depth, ceil(m_height / size), ceil(m_width / size)))

        for n, pad in enumerate(padded):
            for i in range(out.shape[1]):
                for j in range(out.shape[2]):
                    out[n][i][j] = func(
                        pad[i*size:(i+1)*size, j * size:(j+1)*size])

        return out

    def online_train(self, inputs, goal_outputs, epochs):
        n = max(len(inputs), len(goal_outputs))
        order = np.array(range(n))
        for _ in range(epochs):
            np.random.shuffle(order)
            for i in order:
                self.backpropagate(inputs[i], goal_outputs[i])

    def test_guesses(self, inputs, goal_outputs):
        guessed_correctly = 0
        n = max(len(inputs), len(goal_outputs))
        for i, o in zip(inputs, goal_outputs):
            if np.argmax(o) == self.predict(i):
                guessed_correctly += 1
        return guessed_correctly / n


# print(CNN.convolve(np.array([[[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [
#       0, 0, 1, 1, 0], [0, 1, 1, 0, 0]]]), np.array([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]])))
# print(CNN.pool(np.array([[[1, 1, 2, 4, 3], [5, 6, 7, 8, 1], [
#       3, 2, 1, 0, 2], [1, 2, 3, 4, 9]]]), 2, np.max))
cnn = CNN([ConvLayerConfig(8, 5, 2, np.max), ConvLayerConfig(
    8, 3, 2, np.max)], 1, activation_functions.relu, NN(200, 16, 10, 2, activation_functions.sigmoid, activation_functions.dsigmoid, 0.01))

input = np.linspace(0, 1, 28*28).reshape((1, 28, 28))
print(list(map(lambda x: x.shape, cnn.feedforward(input))))
print(cnn.predict(input))
