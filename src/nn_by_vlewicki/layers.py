from enum import Enum
from typing import Callable, Literal
from matplotlib import pyplot as plt
import numpy as np
from .utils import metrics, report_onehot

def relu(x):
    return np.maximum(0, x)

def heaviside(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.clip(np.exp(-x), 0, 500))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class ActivationFunction(Enum):
    RELU = (relu, heaviside)
    SIGMOID = (sigmoid, sigmoid_derivative)


class LossFunction(Enum):
    SQUARES_SUM = (lambda predict, target: np.sum((target - predict)**2), lambda predict, target: predict - target)




class Layer:
    def __init__(self, inputs: int, outputs: int, weights_init: Literal["he", "normal"] = "normal"):
        inputs_with_bias = inputs + 1
        if weights_init == "he":
            self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2 / inputs_with_bias), size=(outputs, inputs_with_bias))
        elif weights_init == "normal":
            self.weights = np.random.normal(size=(outputs, inputs_with_bias))
        self.inputs = None

    def forward(self, inputs):
        self.inputs = np.vstack([np.ones((1, inputs.shape[1])), inputs])
        return np.dot(self.weights, self.inputs)

    def backward(self, layer_error, learning_rate):
        error_of_previous_layer = (self.weights.T @ layer_error)[1:]
        delta = learning_rate * layer_error @ self.inputs.T # / product_error.shape[-1] # product_error.shape[-1] - size of batch
        self.weights = self.weights - delta
        return error_of_previous_layer


class ActivationLayer:
    def __init__(self, activation_function: ActivationFunction = ActivationFunction.RELU):
        self.activation_function = activation_function.value[0]
        self.activation_derivative = activation_function.value[1]

    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        return self.activation_function(inputs)

    def backward(self, layer_error, learning_rate):
        activation_derivative = self.activation_derivative(self.inputs)
        error_of_previous_layer = layer_error * activation_derivative
        return error_of_previous_layer


class BatchNormalizationLayer:
    def __init__(self):
        pass

    def forward(self, inputs: np.ndarray):
        if inputs.shape[-1] == 1:
            return inputs

        shift = np.mean(inputs, axis=-1, keepdims=True)
        scale = np.std(inputs, axis=-1, keepdims=True)
        return (inputs - shift) / (scale + 1e-8)


    def backward(self, activation_error, learning_rate):
        return activation_error

def batched(items: np.ndarray, batch_size: int):
    for i in range(0, items.shape[0], batch_size):
        yield items[i:i+batch_size]

def batched_dataset(inputs, labels, batch_size):
    permutation = np.random.permutation(inputs.shape[-1])
    shuffled_inputs = inputs.T[permutation]
    shuffled_labels = labels.T[permutation]
    for batch_inputs, batch_labels in zip(batched(shuffled_inputs, batch_size), batched(shuffled_labels, batch_size)):
        yield batch_inputs.T, batch_labels.T

class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, error, learning_rate=1.0):
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

class TrainingReport:
    def __init__(self):
        self.epoch = []
        self.accuracy = []
        self.precision = []
        self.recall = []

    def record(self, epoch, accuracy: np.ndarray, precision: np.ndarray, recall: np.ndarray):
        self.epoch.append(epoch)
        self.accuracy.append(accuracy)
        self.precision.append(precision)
        self.recall.append(recall)

    def show(self):
        precision = np.stack(self.precision, axis=-1)
        recall = np.stack(self.recall, axis = -1)
        plt.plot(self.epoch, self.accuracy, label='Accuracy')
        for label, (label_precision, label_recall) in enumerate(zip(precision, recall)):
            plt.plot(self.epoch, label_precision, label=f'Precision #{label}')
            plt.plot(self.epoch, label_recall, label=f'Recall #{label}')
        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

class TrainingInspector:
    def __init__(self, loss_function: Callable[[np.ndarray, np.ndarray], float]):
        self.loss_function = loss_function
        self.epoch = []
        self.loss = []

    def handle_tests_results(self, epoch, predictions, labels):
        self.epoch.append(epoch)
        self.loss.append(self.loss_function(predictions, labels))
        print(self.loss[-1])

    def show(self):
        plt.plot(self.epoch, self.loss, label='Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

class Trainer:
    def __init__(self, train_inputs, train_labels, test_inputs, test_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.test_inputs = test_inputs
        self.test_labels = test_labels

    def fit(self, model:Network, max_epochs=500, batch_size=32, initial_learning_rate=1, loss_function=LossFunction.SQUARES_SUM) -> TrainingInspector:
        loss = loss_function.value[0]
        error = loss_function.value[1]
        epoch = 1
        inspector = TrainingInspector(loss)
        while epoch <= max_epochs:
            learning_rate = initial_learning_rate / (1 + 49 * epoch / max_epochs)
            for batch_inputs, batch_labels in batched_dataset(self.train_inputs, self.train_labels, batch_size):
                predict = model.forward(batch_inputs)
                output_layer_error = error(predict, batch_labels)
                model.backward(output_layer_error, learning_rate=learning_rate)
            test_predict = model.forward(self.test_inputs)
            inspector.handle_tests_results(epoch, test_predict, self.test_labels)
            epoch += 1
        return inspector
