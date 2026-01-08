import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MyPCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.X_scaled = StandardScaler().fit_transform(X)
        cov_matrix = np.cov(self.X_scaled.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        selected_values = eigenvalues[:self.k]
        selected_vectors = eigenvectors[:, :self.k]
        self.matrix = selected_vectors
        return self
        
        

    def transform(self):
        return self.X_scaled.dot(self.matrix)
        
    def fit_transform(self, X):
        return self.fit(X).transform()
