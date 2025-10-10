import numpy as np
import matplotlib.pyplot as plt
import math
# Tanh function :
def tanh(x):
 return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

x = np.linspace(-10, 10, 1000) # 1000 points between -10 and 10
# Calculate the corresponding y-values using the tanh function
y = tanh(x)

# Plot the graph
plt.plot(x, y)

# Add labels and title
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.title("Tanh Function Graph")

# Add a grid for readability
plt.grid(True)

# Display the plot
plt.show()