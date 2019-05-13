import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
import math
import json
from NN import NN


def actifunc(x):
    return 1 / (1 + math.e ** -x)


def dactifunc(y):
    return y * (1 - y)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Creates a neural network for simulating OR function')
    parser.add_argument('train_amount', type=int,
                        help='integer representing the amount of train sessions')
    parser.add_argument('--test', dest='test', action='store_const',
                        const=True, default=False,
                        help='test the NN: <in0> || <in1> == <NN_guess>')
    parser.add_argument('--save', dest='save', action='store_const',
                        const=True, default=False,
                        help='saves the weights in brain.json')
    parser.add_argument('--load', dest='load', default=False,
                        help='loads given brain')
    parser.add_argument('--plot', dest='plot', action='store_const',
                        const=True, default=False,
                        help='plots the NN weights')
    parser.add_argument('-i', dest='interactive', action='store_const',
                        const=True, default=False,
                        help='once done training starts an interactive mode')

    args = parser.parse_args()

    # init
    ornn = NN(2, 2, 1, 2, actifunc, dactifunc, 0.1)

    # load brain
    if args.load:
        with open(args.load) as f:
            ornn.load_brain(json.load(f))

    # training
    for x in range(args.train_amount):
        inputs = [random.randint(0, 1), random.randint(0, 1)]
        goal_output = [int(inputs[0] or inputs[1])]
        ornn.backpropagate(inputs, goal_output)

    # testing
    if args.test:
        for i in range(2):
            for j in range(2):
                print(f'{i} || {j} == {ornn.feedforward([i, j])[0]}')

    # plotting
    if args.plot:
        density = 101
        space = np.linspace(0, 1, density)
        data = [[ornn.feedforward([i, j])[0] for j in space]
                for i in space]
        plt.imshow(data, cmap='gray', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.show()

    # save brain
    if args.save:
        with open('brain.json', 'w') as f:
            json.dump(ornn.serialize(), f)

    # interactive
    if args.interactive:
        print('type bools and get the NN\'s guess! Ctrl+C to exit')
        while True:
            in0 = input('Enter first bool: ').lower() == 'true'
            in1 = input('Enter second bool: ').lower() == 'true'
            out = ornn.feedforward([int(in0), int(in1)])[0]
            print(f'NN guess: {bool(in0)} || {bool(in1)} = {bool(round(out))}')
