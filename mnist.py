import os
import gzip

import numpy as np

__doc__ = """Loader for the MNIST dataset from local files.

Шаг с загрузкой из сети удален, тк ресурс
(http://yann.lecun.com/exdb/mnist/) не доступен.

Файлы будут загружены вручную и сложены в папку:
    mnist_data/
        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz
        t10k-images-idx3-ubyte.gz
        t10k-labels-idx1-ubyte.gz

Источник файлов:
https://ossci-datasets.s3.amazonaws.com/mnist/.
"""

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist_data')


def load_dataset(flatten=False, data_dir=DEFAULT_DATA_DIR):
    """
    Чтение MNIST из локальных .gz файлов
    X_train, y_train, X_val, y_val, X_test, y_test.

    flatten=False -> images shape (N, 28, 28)
    flatten=True  -> images shape (N, 784),
    """

    def _path(filename):
        return os.path.join(data_dir, filename)

    def load_mnist_images(filename):
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(_path(filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return (data / np.float32(256)).squeeze()

    def load_mnist_labels(filename):
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(_path(filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([-1, 28**2])
        X_val = X_val.reshape([-1, 28**2])
        X_test = X_test.reshape([-1, 28**2])

    return X_train, y_train, X_val, y_val, X_test, y_test
