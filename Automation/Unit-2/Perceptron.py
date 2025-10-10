import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Create linearly separable dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)  # convert to -1, +1

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0
lr = 0.1

# Perceptron learning
for epoch in range(20):
    for xi, target in zip(X, y):
        if target * (np.dot(w, xi) + b) <= 0:
            w += lr * target * xi
            b += lr * target

# Plot decision boundary
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title("Perceptron Decision Boundary")
plt.show()
