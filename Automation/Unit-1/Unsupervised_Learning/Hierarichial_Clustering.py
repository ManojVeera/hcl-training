# --- Hierarchical Clustering on Iris Dataset ---

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1️⃣ Load dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)

# 2️⃣ Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Perform Hierarchical Clustering (Ward linkage)
Z = linkage(X_scaled, method='ward')

# 4️⃣ Plot Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (Ward's Method)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# 5️⃣ Assign clusters
clusters = fcluster(Z, t=3, criterion='maxclust')
df['Cluster'] = clusters
df['Actual'] = y

# 6️⃣ Visualize clusters (first 2 features)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette="Set2", s=80)
plt.title("Hierarchical Clustering on Iris Dataset")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

print("\nCluster assignments (first 10):", clusters[:10])
