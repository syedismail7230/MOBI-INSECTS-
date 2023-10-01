import random
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))
            self.biases.append(np.random.randn(layers[i + 1]))

    def forward(self, input):
        output = input
        for i in range(len(self.layers) - 1):
            output = np.dot(output, self.weights[i]) + self.biases[i]
            output = 1.0 / (1.0 + np.exp(-output))

        return output

    def backward(self, input, output, target):
        error = target - output
        gradient = error * (1.0 - output) * output

        self.biases[-1] -= gradient
        self.weights[-1] -= np.dot(output.T, gradient)

        for i in range(len(self.layers) - 2, -1, -1):
            gradient = np.dot(gradient, self.weights[i].T) * (1.0 - output) * output
            self.biases[i] -= gradient
            self.weights[i] -= np.dot(output.T, gradient)

    def train(self, inputs, targets, epochs=1000):
        for i in range(epochs):
            for input, target in zip(inputs, targets):
                output = self.forward(input)
                self.backward(input, output, target)

    def predict(self, input):
        output = self.forward(input)
        return output

def main():
    # Create a neural network with 3 layers:
    # input layer with 2 neurons
    # hidden layer with 5 neurons
    # output layer with 1 neuron
    neural_network = NeuralNetwork([2, 5, 1])

    # Generate some training data
    inputs = np.random.randn(1000, 2)
    targets = np.random.randint(0, 2, 1000)

    # Train the neural network
    neural_network.train(inputs, targets)

    # Generate some test data
    test_inputs = np.random.randn(100, 2)

    # Make predictions on the test data
    predictions = neural_network.predict(test_inputs)

    # Calculate the accuracy
    accuracy = np.sum(predictions == targets) / len(predictions)

    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
