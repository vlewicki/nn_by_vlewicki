from pathlib import Path
from typing import Literal
import numpy as np
from .layers import Layer, Network, Trainer, ActivationFunction, BatchNormalizationLayer, ActivationLayer
from .utils import report_onehot, metrics
from .perceptron import Perceptron


def load_mnist(part: Literal['trn'] | Literal['tst']):
    folder = Path('MNIST')
    x = np.fromfile(folder/f'images_{part}.bin', dtype=np.uint8).reshape(-1, 784).T / 255
    y = np.fromfile(folder/f'labels_{part}.bin', dtype=np.uint8).reshape(1, -1)

    y_onehot = np.zeros((10, y.shape[1]))
    y_onehot[y, np.arange(y.shape[1])] = 1
    return x, y_onehot



def main() -> None:
    x, y = load_mnist('trn')
    x_test, y_test = load_mnist('tst')
    A = x.shape[0]
    B = y.shape[0]
    model = Network([
        Layer(A, 128),
        Layer(128, 32),
        Layer(32, B),
    ])
    permutation = np.random.permutation(x.shape[1])
    for i in range(300):
        x_epoch = x[:, permutation]
        y_epoch = y[:, permutation]
        batch_size = 5000
        # print('Weight', model.layers[-1].weights.max(), model.layers[-1].weights.min())
        for j in range(0, x_epoch.shape[1], batch_size):
            x_batch = x_epoch[:, j:j+batch_size]
            y_batch = y_epoch[:, j:j+batch_size]
            predict = model.forward(x_batch)
            error = predict - y_batch
            # if j == 0:
                # print('Error #0 batch', error.max(), error.min())
            model.backward(error, learning_rate=0.01)

        if i % 10 == 0:
            print('Epoch', i)
            test_predict = model.forward(x_test)
            report_onehot(test_predict, y_test)
        # report(predict, y)
