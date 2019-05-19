import json
import gzip
import argparse
import numpy as np
import webbrowser
import simple_http_server.server as server
from simple_http_server import request_map
from simple_http_server import StaticFile
from NN import NN


def load_dataset(image_data_path, label_data_path, to_pixels, to_labels, amount):
    f = gzip.open(image_data_path, "rb")
    l = gzip.open(label_data_path, "rb")

    f.read(16)
    l.read(8)

    for i in range(amount):
        to_labels.append(ord(l.read(1)))
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        to_pixels.append(image)

    f.close()
    l.close()


def map_labels(position):
    empty = np.array(range(10))*0
    empty[position] = 1
    return empty


def map_pixels(pixel_array):
    return list(map(lambda x: x/255, pixel_array))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates a NN for the classical mnist dataset (without a convolutional layer)')
    parser.add_argument('train_amount', type=int,
                        help='integer representing the amount of training sessions')
    parser.add_argument('--test-every', dest='test_every', default=False, type=int,
                        help='prints out the accuracy every x trains')
    parser.add_argument('--save', dest='save', action='store_const',
                        const=True, default=False,
                        help='saves the weights in brain.json')
    parser.add_argument('--load', dest='load', default=False,
                        help='loads given brain')
    parser.add_argument('-i', dest='interactive', action='store_const',
                        const=True, default=False,
                        help='once done training starts an interactive mode')

    args = parser.parse_args()

    train_pixels = []
    train_labels = []
    test_pixels = []
    test_labels = []

    sessions = []
    if args.test_every:
        n = args.train_amount
        while True:
            if n - args.test_every <= 0:
                sessions.append(n)
                break
            else:
                sessions.append(args.test_every)
                n -= args.test_every
    else:
        sessions.append(args.train_amount)

    print('loading data...')
    load_dataset("train_pixels.gz", "train_labels.gz",
                 train_pixels, train_labels, 60000)
    load_dataset("test_pixels.gz", "test_labels.gz",
                 test_pixels, test_labels, 10000)

    mnistnn = NN.from_config('config.yaml')
    if args.load:
        with open(args.load) as f:
            mnistnn.load_brain(json.load(f))

    mapped_train_pixels = list(map(map_pixels, train_pixels))
    mapped_train_labels = list(map(map_labels, train_labels))
    mapped_test_pixels = list(map(map_pixels, test_pixels))
    mapped_test_labels = list(map(map_labels, test_labels))

    print('training...')
    for sess in sessions:
        mnistnn.online_train(mapped_train_pixels, mapped_train_labels, sess)
        print(mnistnn.test_guesses(mapped_test_pixels,
                                   mapped_test_labels, 1000)*100, '%')

    if args.save:
        with open('brain.json', 'w') as f:
            json.dump(mnistnn.serialize(), f)

    if args.interactive:
        @request_map('/')
        def get_root():
            return StaticFile('./index.html', 'text/html; charset=utf-8')

        @request_map('/favicon.ico')
        def favicon():
            return StaticFile('./favicon.ico', 'image/x-icon')

        @request_map('/guess', method='POST')
        def take_guess(pixels):
            inputs = map_pixels([int(x) for x in pixels.split('v')])
            return {'guess': int(np.argmax(mnistnn.feedforward(inputs)))}

        webbrowser.open('http://localhost:9090')
        server.start()
