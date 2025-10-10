# Import required libraries :
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function :
def sigmoid(x):
 return 1/(1+np.exp(-x))

ip = -1
op = sigmoid(ip)
print(op)

# Define a range of x-values
x = np.linspace(-100, 100, 1000) # 1000 points between -10 and 10

# Calculate the corresponding y-values using the sigmoid function
y = sigmoid(x)

# Plot the graph
plt.plot(x, y)

# Add labels and title
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.title("Sigmoid Function Graph")

# Add a grid for readability
plt.grid(True)

# Display the plot
plt.show()