import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. Load dataset ---
data = fetch_california_housing(as_frame=True)
X = data.data.values      # convert to numpy array
y = data.target.values.reshape(-1, 1)  # column vector

# --- 2. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Add bias term (intercept) ---
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias  = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# --- 4. Compute weights using Normal Equation ---
# w = (X^T X)^(-1) X^T y
w = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

# --- 5. Predict ---
y_pred = X_test_bias @ w

# --- 6. Evaluation ---
mse = np.mean((y_test - y_pred)**2)
r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

print("MSE:", mse)
print("R²:", r2)

# --- 7. Plot ---
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression (NumPy)")
plt.show()
