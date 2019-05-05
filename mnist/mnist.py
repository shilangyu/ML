import gzip
import argparse
import numpy as np
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
    args = parser.parse_args()

    train_pixels = []
    train_labels = []
    test_pixels = []
    test_labels = []

    load_dataset("train_pixels.gz", "train_labels.gz",
                 train_pixels, train_labels, 5000)
    load_dataset("test_pixels.gz", "test_labels.gz",
                 test_pixels, test_labels, 5000)

    mnistnn = NN(28*28, 16, 10, 2,
                 lambda x: 1 / (1 + np.exp(-x)), lambda y: y * (1-y), 0.1)

    mapped_train_pixels = list(map(map_pixels, train_pixels))
    mapped_train_labels = list(map(map_labels, train_labels))

    mnistnn.train(mapped_train_pixels, mapped_train_labels, args.train_amount)
