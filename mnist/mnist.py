import gzip
import sys


def load_dataset(image_data_path, label_data_path, to, amount):
    f = gzip.open(image_data_path, "rb")
    l = gzip.open(label_data_path, "rb")

    f.read(16)
    l.read(8)

    for i in range(amount):
        to.append({'label': ord(l.read(1))})
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        to[-1]['pixels'] = image

    f.close()
    l.close()


train_data = []
test_data = []

load_dataset("train_images.gz", "train_labels.gz", train_data, 60000)
load_dataset("test_images.gz", "test_labels.gz", test_data, 10000)
print('done loading data')
