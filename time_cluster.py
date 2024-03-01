import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Example time series data (replace this with your own data)
time_series_list = [
    np.sin(np.linspace(0, 4 * np.pi, 100)),
    np.cos(np.linspace(0, 6 * np.pi, 150)),
    np.tan(np.linspace(0, 2 * np.pi, 120)),
    # Add more time series as needed
]

# Calculate pairwise distances using DTW
distance_matrix = np.zeros((len(time_series_list), len(time_series_list)))
for i, ts1 in enumerate(time_series_list):
    for j, ts2 in enumerate(time_series_list):
        _, path = fastdtw(ts1, ts2, dist=euclidean)
        distance_matrix[i, j] = len(path)

# Visualize the distance matrix
plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
plt.title('DTW Distance Matrix')
plt.colorbar()
plt.show()

# Apply K-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(distance_matrix)

# Print cluster assignments
for i, label in enumerate(cluster_labels):
    print(f"Time series {i+1} belongs to Cluster {label + 1}")
