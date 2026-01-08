import numpy as np

def confusion_matrix(y_pred, y_test):
    n = len(y_pred)
    matrix = np.array([[0, 0], [0, 0]])
    for i in range(n):
        if y_pred[i] == 1 and y_pred[i] == y_test[i]:
            matrix[0][0] += 1
        elif y_pred[i] == 1 and y_pred[i] != y_test[i]:
            matrix[0][1] += 1
        elif y_pred[i] == 0 and y_pred[i] != y_test[i]:
            matrix[1][0] += 1
        else:
            matrix[1][1] += 1
    return matrix

    
def accuracy(y_pred, y_test):
    A = confusion_matrix(y_pred, y_test)
    return (A[0][0] + A[1][1]) / (A[0][0] + A[0][1] + A[1][0] + A[1][1])


def precision(y_pred, y_test):
    A = confusion_matrix(y_pred, y_test)
    return A[0][0] / (A[0][0] + A[0][1])


def recall(y_pred, y_test):
    A = confusion_matrix(y_pred, y_test)
    return A[0][0] / (A[0][0] + A[1][0])


def f1(y_pred, y_test):
    pr = precision(y_pred, y_test)
    rec = recall(y_pred, y_test)
    return 2 * (pr * rec) / (pr + rec)
