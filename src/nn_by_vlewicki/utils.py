import numpy as np


def metrics(predict: np.ndarray, target: np.ndarray, threshold=0.5) -> tuple[float, np.ndarray, np.ndarray]:
    if predict.shape[0] == 1: # binary classification
        accuracy = np.mean((target >= threshold) == (predict >= threshold), axis=-1)
        precision = np.mean(predict[target >= threshold] >= threshold, axis=-1)
        recall = np.mean(target[predict >= threshold] >= threshold, axis=-1)
        return accuracy, np.array([precision]), np.array([recall])
    else:
        n_classes = predict.shape[0]
        target = target.argmax(axis=0)
        predict = predict.argmax(axis=0)
        accuracy = np.mean(target == predict, axis=-1)
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        for label in range(n_classes):
            precision[label] = np.mean(predict[target == label] == label)
            recall[label] = np.mean(target[predict == label] == label)
        return accuracy, precision, recall


def report_onehot(y_predict, y_target):
    y_target = y_target.argmax(axis=0)
    y_predict = y_predict.argmax(axis=0)
    print(f'Accuracy: {np.mean(y_predict == y_target)}')
    print("Label\tPrecision\tRecall")

    for label in np.unique(y_target):
        print(f'Label {label}:\t{np.mean(y_predict[y_target == label] == label)}', end='\t')
        print(np.mean(y_target[y_predict == label] == label))
