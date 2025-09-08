import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load and scale
digits = load_digits()
X_scaled = StandardScaler().fit_transform(digits.data)

# PCA for visualization
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# KMeans
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X_scaled)
clusters = kmeans.labels_

# Plot
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="tab10", s=20)
plt.scatter(PCA(n_components=2).fit(X_scaled).transform(kmeans.cluster_centers_)[:,0],
            PCA(n_components=2).fit(X_scaled).transform(kmeans.cluster_centers_)[:,1],
            c="red", marker="X", s=200, label="Centroids")
plt.title("K-Means on Digits (PCA 2D)")
plt.legend()
plt.show()
