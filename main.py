import numpy as np


class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.uniform(-1, 1, size=num_inputs)
        self.bias = np.random.uniform(-1, 1)

    def activate(self, inputs):
        self.inputs = inputs
        z = np.dot(inputs, self.weights) + self.bias
        self.output = 1 / (1 + np.exp(-z))  # sigmoid
        return self.output

    def sigmoid_derivative(self):
        # derivative of sigmoid
        return self.output * (1 - self.output)

    def update_weights(self, delta, lr):
        # adjust weights and bias
        self.weights += lr * delta * self.inputs
        self.bias += lr * delta

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def backward(self, errors, lr):
        # compute deltas and propagate gradients back
        deltas = []
        for i, neuron in enumerate(self.neurons):
            delta = errors[i] * neuron.sigmoid_derivative()
            neuron.update_weights(delta, lr)
            deltas.append(delta)
        # return error for previous layer
        return np.dot(np.array([n.weights for n in self.neurons]).T, deltas)

class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.1, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        self.layers = []
        # create network
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i+1], layer_sizes[i]))

    def train(self, inputs, targets):
        for epoch in range(self.epochs):
            total_error = 0
            for x, y in zip(inputs, targets):
                # forward pass
                activations = [x]
                for layer in self.layers:
                    activations.append(layer.forward(activations[-1]))

                # compute error (MSE part)
                output_error = y - activations[-1]
                total_error += np.sum(output_error ** 2)

                # backward pass
                error = output_error
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.lr)

            if epoch % 1000 == 0:
                avg_error = total_error / len(inputs)
                print(f"Epoch {epoch}, MSE: {avg_error}")

    def predict(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

# XOR truth table
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([[0],   [1],   [1],   [0]])

# network: 2 inputs → 2 hidden → 1 output
nn = NeuralNetwork([2, 2, 1], lr=0.1, epochs=10000)

nn.train(inputs, outputs)

# Test predictions
for color, x in zip(["00","01","10","11"], inputs):
    pred = nn.predict(x)
    print(f"{x} → {pred.round(3)}")
