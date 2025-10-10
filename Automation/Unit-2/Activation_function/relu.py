import numpy as np
import matplotlib.pyplot as plt

# ReLU function :
def relu(x):
    return np.maximum(0, x)

# Input values (x-axis)
x = np.linspace(-10, 10, 1000)  # 1000 points between -10 and 10

# Bias term
bias = 0.01

# Apply bias (you can treat weights as slope = 1 for visualization)
temp = x + bias

# Calculate the corresponding y-values using the ReLU function
y = relu(temp)

# Plot the graph
plt.plot(x, y, label="ReLU(x + bias)")

# Add labels and title
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.title("ReLU Function Graph")

# Add grid and legend
plt.grid(True)
plt.legend()

# Display the plot
plt.show()
