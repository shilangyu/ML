from NN import NN
import math
import random
import argparse
import json


def actifunc(x):
    return 1 / (1 + math.e ** -x)


def dactifunc(y):
    return y * (1 - y)


class dataline:
    def __init__(self, string):
        first, operand, second, result = string[0], string[1], string[2], string[4:]

        self.first = float(first)
        self.second = float(second)
        self.result = float(result)
        self.operand = operand

    def mapped_inputs(self):
        first = (self.first - 1) / 8
        operand = ['+', '-', '*', '/'].index(self.operand) / 3
        second = (self.second - 1) / 8

        return [first, operand, second]

    def mapped_result(self):
        if self.operand == '+':
            return [(self.first + self.second - 2) / 16]
        if self.operand == '-':
            return [(self.first - self.second + 8) / 16]
        if self.operand == '*':
            return [(self.first * self.second - 1) / 80]
        if self.operand == '/':
            return [(self.first / self.second - 1/9) / (9 - 1/9)]

    def unmap_output(self, output):
        if self.operand == '+':
            return output * 16 + 2
        if self.operand == '-':
            return output * 16 - 8
        if self.operand == '*':
            return output * 80 + 1
        if self.operand == '/':
            return output * (9 - 1/9) + 1/9

    def tostring(self):
        return f'{int(self.first)}{self.operand}{int(self.second)}={self.result}'


def unmap(x, sign):
    if sign == '*':
        return x*81
    elif sign == '+':
        return x*18
    elif sign == '-':
        return (x-.5)*18


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates a neural network that learns how to perform [/*-+] calculations')
    parser.add_argument('train_amount', type=int,
                        help='integer representing the amount of train sessions')
    parser.add_argument('--gen-file', dest='gen_file', action='store_const',
                        const=True, default=False,
                        help='creates outs.txt with NN guesses: <in0><operand><in1>=<correct> | <NN_guess>')
    parser.add_argument('--save', dest='save', action='store_const',
                        const=True, default=False,
                        help='saves the weights in brain.json')
    parser.add_argument('--load', dest='load', default=False,
                        help='loads given brain')

    args = parser.parse_args()

    # init
    calcnn = NN(3, 10, 1, 3, actifunc, dactifunc, 0.1)

    # load brain
    if args.load:
        with open(args.load) as f:
            calcnn.load_brain(json.load(f))

    # data
    traindata = []
    testdata = []
    with open('./train.txt') as f:
        for line in f.readlines():
            traindata.append(dataline(line))
    with open('./test.txt') as f:
        for line in f.readlines():
            testdata.append(dataline(line))

    # training
    inputs = list(map(lambda x: x.mapped_inputs(), traindata))
    goal_ouputs = list(map(lambda x: x.mapped_result(), traindata))
    calcnn.train(inputs, goal_ouputs, args.train_amount)

    # testing
    inputs = list(map(lambda x: x.mapped_inputs(), testdata))
    goal_ouputs = list(map(lambda x: x.mapped_result(), testdata))
    acc = calcnn.test(inputs, goal_ouputs, 1000)

    print(f'Accuracy: {acc*100}%')

    # testing with output to file
    if args.gen_file:
        guesses = []
        for data in testdata:
            inputs = data.mapped_inputs()
            goal_output = data.mapped_result()[0]
            output = calcnn.feedforward(inputs)[0]

            guesses.append(f'{data.tostring()} | {data.unmap_output(output)}')

        with open('outs.txt', mode='w') as f:
            f.write('\n'.join(guesses))

    # save brain
    if args.save:
        with open('brain.json', 'w') as f:
            json.dump(calcnn.serialize(), f)
