import numpy as np

class MyKMeans:
    def __init__(self, n_clusters=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        # Случайная инициализация центров
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        centers = X[indices]

        for _ in range(self.max_iter):
            # Присваиваем каждой точке ближайший центр
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Пересчитываем центры как среднее точек в каждом кластере
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Проверка на сходимость (если центры не меняются)
            if np.allclose(centers, new_centers):
                break

            centers = new_centers

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self
