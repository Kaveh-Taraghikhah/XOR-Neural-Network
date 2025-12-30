import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

W1 = np.random.uniform(-1, 1, (2, 2))
b1 = np.zeros((1, 2))
W2 = np.random.uniform(-1, 1, (2, 1))
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

lr = 0.3
epochs = 30000

for _ in range(epochs):
    h = sigmoid(X @ W1 + b1)
    y_hat = sigmoid(h @ W2 + b2)

    error = y - y_hat
    d_out = error * sigmoid_deriv(y_hat)
    d_hidden = d_out @ W2.T * sigmoid_deriv(h)

    W2 += lr * h.T @ d_out
    b2 += lr * d_out.sum(axis=0)
    W1 += lr * X.T @ d_hidden
    b1 += lr * d_hidden.sum(axis=0)

print(y_hat.round(3))
