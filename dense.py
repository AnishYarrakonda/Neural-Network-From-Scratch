# imports
import numpy as np

# Hidden Layer
class Dense:

    # constructs a hidden layer
    def __init__(self, num_inputs, num_outputs):
        self.inputs = None                                                  # stores inputs for later
        self.weights = np.random.randn(num_inputs, num_outputs) * 0.1       # randomly initialize weights
        self.biases = np.zeros((1, num_outputs))                            # biases are set to zero

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs                                    # cache inputs
        outputs = inputs @ self.weights + self.biases           # calculate outputs using dot product and sum with biases
        return outputs                                          # return outputs for the next layer