import numpy as np

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Perceptron function with ReLU
def perceptron(x1, x2, weights, bias):
    linear_output = weights[0]*x1 + weights[1]*x2 + bias
    activated_output = relu(linear_output)
    return 1 if activated_output > 0 else 0  # Threshold to binary output

# Define weights and bias for AND gate
weights = [1, 1]
bias = -1.5  # ensures only (1,1) gives output 1

# Test all input combinations
inputs = [(0,0), (0,1), (1,0), (1,1)]
print("AND Gate using Perceptron (ReLU):")
for x1, x2 in inputs:
    print(f"{x1} AND {x2} = {perceptron(x1, x2, weights, bias)}")
