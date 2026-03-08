# imports
from abc import ABC, abstractmethod
import numpy as np

# activation interface
class Activation(ABC):

    # must have a forward pass method
    @abstractmethod
    def forward(self, inputs):
        pass

    # must have a backpropagation method
    @abstractmethod
    def backward(self, dvalues):
        pass


# Rectified Linear Unit Activation Function (ReLU)
class ReLU(Activation):

    # constructs a ReLU activation layer
    def __init__(self):
        self.inputs = None                      # cache inputs

    # forward pass (clips values at zero)
    def forward(self, inputs):
        self.inputs = inputs                    # cache inputs
        return np.maximum(0, inputs)        # return clipped values

    # backward pass
    def backward(self, dvalues):
        raise NotImplementedError("ReLU.backward is not implemented yet.")

# Sigmoid Activation Function
class Sigmoid(Activation):

    # constructs a Sigmoid activation layer
    def __init__(self):
        self.inputs = None                      # cache inputs

    # forward pass (applies sigmoid function)
    def forward(self, inputs):
        self.inputs = inputs                    # cache inputs
        return 1 / (1 + np.exp(-inputs))        # apply the function and return outputs

    # backward pass
    def backward(self, dvalues):
        raise NotImplementedError("Sigmoid.backward is not implemented yet.")

# Hyperbolic Tangent Activation Function (tanh)
class Tanh(Activation):

    # constructs a tanh activation layer
    def __init__(self):
        self.inputs = None

    # forward pass (applies tanh function)
    def forward(self, inputs):
        self.inputs = inputs                    # cache inputs
        return np.tanh(inputs)                  # apply the function and return outputs

    # backward pass
    def backward(self, dvalues):
        raise NotImplementedError("Tanh.backward is not implemented yet.")

# Softmax Activation Function (for logits in output layer only)
class Softmax(Activation):

    # constructs a Softmax layer
    def __init__(self):
        self.inputs = None                      # cache inputs

    # forward pass (normalizes logits into exponent-based scores)
    def forward(self, inputs):
        self.inputs = inputs                                            # cache raw input logits
        small_inputs = inputs - np.max(inputs, axis=1, keepdims=True)   # shift each row by its max value to prevent huge exponents
        exp_values = np.exp(small_inputs)                               # exponentiate shifted logits
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)   # apply our formula to return the probability distribution

    # backward pass
    def backward(self, dvalues):
        raise NotImplementedError("Softmax.backward is not implemented yet.")

# Leaky ReLU (small negative values instead of 0)
class LeakyReLU(Activation):

    # constructs a Leaky ReLU layer
    def __init__(self, alpha=0.01):
        self.inputs = None                      # cache inputs
        self.alpha = alpha                      # slope

    # forward pass (decreases the magnitude of negative values)
    def forward(self, inputs):
        self.inputs = inputs                                        # cache inputs
        return np.where(inputs > 0, inputs, self.alpha * inputs)    # multiply negative numbers by our slope and return outputs

    # backward pass
    def backward(self, dvalues):
        raise NotImplementedError("LeakyReLU.backward is not implemented yet.")
