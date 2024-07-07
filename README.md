# Clustering Evaluation with Silhouette Score and Davies-Bouldin Index

## Purpose of the Code

This repository provides an example of how to use the Silhouette Score and Davies-Bouldin Index to evaluate the performance of a clustering algorithm. The example demonstrates the application of these metrics on a dataset with overlapping clusters to illustrate their effectiveness in assessing clustering quality.

## Metrics to Evaluate Clustering Models

### 1. Silhouette Score

The Silhouette Score measures how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1:
- **+1:** The sample is far away from the neighboring clusters.
- **0:** The sample is on or very close to the decision boundary between two neighboring clusters.
- **-1:** The sample is assigned to the wrong cluster.

**Calculation:**
1. Compute the average distance between a sample and all other points in the same cluster (a).
2. Compute the average distance between a sample and all points in the nearest cluster (b).
3. The silhouette score for a sample is given by:
\[ \text{silhouette} = \frac{b - a}{\max(a, b)} \]

### 2. Davies-Bouldin Index

The Davies-Bouldin Index measures the average similarity ratio of each cluster with its most similar cluster. It is defined as the average ratio of within-cluster distances to between-cluster distances:
- **Lower values:** Indicate better clustering performance.

**Calculation:**
1. For each cluster, compute the average distance between each point in the cluster and the cluster centroid (within-cluster scatter).
2. For each pair of clusters, compute the distance between the centroids (between-cluster distance).
3. Compute the Davies-Bouldin Index as the average of the maximum ratios of within-cluster scatter to between-cluster distance for each cluster.

## Example Code

Below is the Python code to generate an example dataset with overlapping clusters and evaluate it using the Silhouette Score and Davies-Bouldin Index.
The `demo.ipynb` file includes a similar use example.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Generate example data with more ambiguous clusters
X, y = make_blobs(n_samples=500, centers=4, cluster_std=2.5, random_state=42)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Calculate silhouette score
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score}")

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_index}")

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.title(f'Silhouette Score: {sil_score}, Davies-Bouldin Index: {db_index}')
plt.show()
