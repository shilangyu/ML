import math
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
from NN import NN


def actifunc(x):
    return 1 / (1 + math.e ** -x)


def dactifunc(y):
    return y * (1 - y)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Creates a neural network for simulating XOR function')
    parser.add_argument('train_amount', type=int,
                        help='integer representing the amount of train sessions')
    parser.add_argument('--test', dest='test', action='store_const',
                        const=True, default=False,
                        help='test the NN: <in0> ^ <in1> == <NN_guess>')
    parser.add_argument('--plot', dest='plot', action='store_const',
                        const=True, default=False,
                        help='plots the NN weights')

    args = parser.parse_args()

    # init
    xornn = NN(2, 10, 1, 3, actifunc, dactifunc, 0.1)

    # training
    for x in range(args.train_amount):
        input = [random.randint(0, 1), random.randint(0, 1)]
        goal_output = [input[0] ^ input[1]]
        xornn.backpropagate(input, goal_output)

    # testing
    if args.test:
        for i in range(2):
            for j in range(2):
                print(f'{i} ^ {j} == {xornn.feedforward([i, j])[0]}')

    # plotting
    if args.plot:
        density = 101
        space = np.linspace(0, 1, density)
        data = [[xornn.feedforward([i, j])[0] for j in space]
                for i in space]
        plt.imshow(data, cmap='gray', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.show()
