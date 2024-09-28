import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, init_method='random', max_iter=100):
        """
        Initialize the KMeans clustering algorithm.

        Parameters:
        - n_clusters: Number of clusters.
        - init_method: Initialization method ('random', 'farthest', 'kmeans++', 'manual').
        - max_iter: Maximum number of iterations.
        """
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter

        self.centroids = None
        self.labels = None

    def fit(self, X, initial_centroids=None):
        """
        Compute KMeans clustering.

        Parameters:
        - X: Data points (numpy array of shape (n_samples, n_features)).
        - initial_centroids: Initial centroids for 'manual' initialization.
        """
        self.X = X

        # Initialize centroids based on the selected method
        if self.init_method == 'random':
            self.centroids = self.initialize_random()
        elif self.init_method == 'farthest':
            self.centroids = self.initialize_farthest()
        elif self.init_method == 'kmeans++':
            self.centroids = self.initialize_kmeans_plus_plus()
        elif self.init_method == 'manual':
            if initial_centroids is not None:
                self.centroids = np.array(initial_centroids)
            else:
                raise ValueError("Initial centroids must be provided for manual initialization.")
        else:
            raise ValueError("Invalid initialization method.")

    def initialize_random(self):
        """Randomly select k data points as initial centroids."""
        indices = np.random.choice(self.X.shape[0], self.n_clusters, replace=False)
        return self.X[indices]

    def initialize_farthest(self):
        """Initialize centroids to be as far apart as possible."""
        centroids = []

        # Randomly choose the first centroid
        idx = np.random.randint(self.X.shape[0])
        centroids.append(self.X[idx])

        # Select remaining centroids
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.sum((x - c)**2) for c in centroids]) for x in self.X])
            next_idx = np.argmax(dist_sq)
            centroids.append(self.X[next_idx])

        return np.array(centroids)

    def initialize_kmeans_plus_plus(self):
        """Initialize centroids using the KMeans++ algorithm."""
        centroids = []

        # Randomly choose the first centroid
        idx = np.random.randint(self.X.shape[0])
        centroids.append(self.X[idx])

        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.sum((x - c)**2) for c in centroids]) for x in self.X])
            probabilities = dist_sq / dist_sq.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()

            for idx, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids.append(self.X[idx])
                    break

        return np.array(centroids)

    def assign_clusters(self):
        """Assign each data point to the nearest centroid."""
        distances = np.array([[np.linalg.norm(x - c) for c in self.centroids] for x in self.X])
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self):
        """Update centroids by calculating the mean of assigned data points."""
        centroids = []
        for i in range(self.n_clusters):
            points = self.X[self.labels == i]
            if len(points) > 0:
                centroids.append(points.mean(axis=0))
            else:
                # Handle empty clusters by reinitializing the centroid randomly
                centroids.append(self.X[np.random.randint(0, self.X.shape[0])])
        return np.array(centroids)

    def predict(self, X):
        """Predict the nearest cluster each sample in X belongs to."""
        distances = np.array([[np.linalg.norm(x - c) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def step(self):
        self.labels = self.assign_clusters()
    
        # Update step
        old_centroids = self.centroids.copy()
        self.centroids = self.update_centroids()
    
        # Check for convergence
        return np.allclose(self.centroids, old_centroids)

    
    
