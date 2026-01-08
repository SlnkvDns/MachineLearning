import numpy as np
import pandas as pd
from scipy import stats

import numpy as np
from scipy import stats

class KNN:
    def __init__(self, k):
        self.k = k
        

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
            

    def predict(self, X_test):
        predicted_classes = np.zeros(X_test.shape[0])
        m = 0
        for obj in X_test:
            predicted_obj = np.array(obj)
            distances = self.calculate_distance(predicted_obj, self.X_train)
            sorted_distances = sorted(distances, key=lambda x: x[1])
            nearest_obj = []
            for i in range(self.k):
                nearest_obj.append(sorted_distances[i][0])
            neighbors_classes = [self.y_train[i] for i in nearest_obj]
            predicted_class = stats.mode(neighbors_classes)[0]
            predicted_classes[m] = predicted_class
            m += 1
        return predicted_classes
            

    def calculate_distance(self, point, X_train):
        distances = []
        for i in range(X_train.shape[0]):
            obj = np.array(X_train[i])
            dist = np.linalg.norm(point - obj)
            distances.append([i, dist])
        return distances
  