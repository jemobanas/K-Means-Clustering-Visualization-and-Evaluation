import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from matplotlib.animation import FuncAnimation

# Generate synthetic unlabeled data with more scattered clusters
X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=5.0, random_state=80)

# Function to apply K-Means clustering with animation of centroid adjustments
def animate_kmeans_iterations(X, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, max_iter=1, random_state=80)
    kmeans.fit(X)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Assign colors for each cluster manually: yellow, red, blue
    colors = ['yellow', 'red', 'blue']

    scatter = ax.scatter(X[:, 0], X[:, 1], c='gray', s=50, alpha=0.6, label='Data Points')
    centroids_scatter = ax.scatter([], [], c='black', s=200, alpha=0.8, marker='*', label='Centroids')  # Black stars

    # Store centroids for each iteration
    centroid_history = [kmeans.cluster_centers_]

    # Function to update the plot for each iteration
    def update(i):
        nonlocal kmeans
        # Reinitialize KMeans for each iteration to get a fresh start
        kmeans = KMeans(n_clusters=n_clusters, init=centroid_history[-1] if i > 0 else 'k-means++', n_init=1, max_iter=1, random_state=80)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_

        # Update the centroid scatter plot
        centroids_scatter.set_offsets(centroids)
        centroid_history.append(centroids)

        # Update point colors based on new labels
        labels = kmeans.labels_
        cluster_colors = [colors[label] for label in labels]
        scatter.set_color(cluster_colors)

        ax.set_title(f'Iteration {i}')
        return scatter, centroids_scatter

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(1, 25 + 1), interval=500, repeat=False)

    ax.set_title('K-Means Clustering Animation')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

    plt.show()

    return kmeans

# Determine the optimal number of clusters using the elbow method
inertia_values = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=80)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Optimal number of clusters (from elbow method)
optimal_k = 3

# Animate K-Means iterations
kmeans = animate_kmeans_iterations(X, optimal_k)

# Evaluate clustering performance
labels = kmeans.labels_
silhouette = silhouette_score(X, labels)
davies_bouldin = davies_bouldin_score(X, labels)
random_labels = np.random.randint(0, optimal_k, size=len(labels))
ari = adjusted_rand_score(random_labels, labels)

# Print performance metrics
print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
print(f"Adjusted Rand Index (ARI): {ari:.2f}")

# Final visualization of the clusters with assigned colors
plt.figure(figsize=(8, 5))

# Assign colors to clusters manually
final_cluster_colors = ['yellow', 'red', 'blue']
final_cluster_labels = [final_cluster_colors[label] for label in labels]  # Corrected line

# Scatter plot with cluster colors
plt.scatter(X[:, 0], X[:, 1], c=final_cluster_labels, s=50, alpha=0.6)

# Plot final centroids
final_centroids = kmeans.cluster_centers_
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='black', s=200, alpha=0.8, marker='*', label='Final Centroids')

plt.title('Final K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
