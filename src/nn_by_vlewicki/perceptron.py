import numpy as np

def sign(x):
    return np.where(x >= 0, 1, -1)

class Perceptron:
    def __init__(self, n):
        self.learning_rate = 1
        self.weights = np.random.randn(n)

    def forward(self, x):
        return sign(np.dot(self.weights, x))

    def backward(self, x, y):
        if y * self.forward(x) < 0:
            self.weights += (self.learning_rate * y) * x
