import math
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
from NN import NN


def actifunc(x):
    return 1 / (1 + math.e ** -x)


def dactifunc(y):
    return y * (1 - y)


if __name__ == '__main__':
    if '--help' in sys.argv:
        print(''' 
        train   Trains the NN
        --test  Run tests [got, correct]
        --plot  Plot the NN
        ''')
        sys.exit()

    # init
    xornn = NN(2, 10, 1, 3, actifunc, dactifunc, 0.1)

    # training
    if 'train' in sys.argv:
        for x in range(100000):
            input = [random.randint(0, 1), random.randint(0, 1)]
            goal_output = [input[0] ^ input[1]]
            xornn.backpropagate(input, goal_output)

    # testing
    if '--test' in sys.argv:
        for i in range(2):
            for j in range(2):
                print(xornn.feedforward([i, j]), int(i ^ j))

    # plotting
    if '--plot' in sys.argv:
        density = 101
        space = np.linspace(0, 1, density)
        data = [[xornn.feedforward([i, j])[0] for j in space]
                for i in space]
        plt.imshow(data, cmap='gray', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.show()
