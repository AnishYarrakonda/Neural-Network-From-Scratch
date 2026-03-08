# imports
from dense import Dense
from activations import *
from loss import *

# base class for our neural network
class NNModel:

    # constructs a model (starts empty -> no layers)
    def __init__(self, lr=0.001):
        self.layers = []                        # stores the layers
        self.loss = None                        # stores loss function
        self.lr = lr                            # stores the learning rate
        self.training = True                    # training mode flag (controls dropout behaviour)

    # adds a layer
    def add(self, layer):
        self.layers.append(layer)               # add any layer type (dense, activation, etc.)

    # sets the loss function
    def set_loss(self, loss):
        self.loss = loss                        # use an appropriate loss function

    # forward pass
    def forward(self, X):
        for layer in self.layers:                       # loop through each layer
            if isinstance(layer, Dropout):              # dropout needs to know if we are training or not
                X = layer.forward(X, self.training)     # pass the training flag through
            else:                                       # otherwise
                X = layer.forward(X)                    # update the inputs through each layer's forward pass
        return X                                        # return the final outputs

    # backpropagation
    def backward(self, y_pred, y_true, lr):
        error = self.loss.backward(y_pred, y_true)  # backwards (loss function)
        for layer in reversed(self.layers):         # go in reverse order of layers
            if isinstance(layer, Dense):            # if Dense layer
                error = layer.backward(error, lr)   # decrease step size using learning rate
            else:                                   # otherwise
                error = layer.backward(error)       # just go backward