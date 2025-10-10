# Experiment 1: Implement and visualize activation functions
import numpy as np
import matplotlib.pyplot as plt

# Define activations and derivatives
def sigmoid(x): return 1 / (1 + np.exp(-x))
def dsigmoid(x): s = sigmoid(x); return s * (1 - s)

def tanh(x): return np.tanh(x)
def dtanh(x): return 1 - np.tanh(x) ** 2

def relu(x): return np.maximum(0, x)
def drelu(x): return (x > 0).astype(float)

def leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)
def dleaky_relu(x, a=0.01): return np.where(x > 0, 1, a)

x = np.linspace(-5, 5, 400)

# Plot each activation and derivative
funcs = [
    ("Sigmoid", sigmoid, dsigmoid),
    ("Tanh", tanh, dtanh),
    ("ReLU", relu, drelu),
    ("Leaky ReLU", leaky_relu, dleaky_relu)
]

plt.figure(figsize=(10, 8))
for i, (name, f, df) in enumerate(funcs, 1):
    plt.subplot(2, 2, i)
    plt.plot(x, f(x), label=f'{name}')
    plt.plot(x, df(x), '--', label=f'{name} Derivative')
    plt.title(name)
    plt.legend()
plt.tight_layout()
plt.show()
