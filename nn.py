"""
A class for a simple feed-forward neural network with one hidden layer.
The code is adapted from `Make Your Own Neural Network` by Tariq Rashid.
ORIGINAL: https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork
BOOK: https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G/
"""

import numpy as np
import scipy.special


class neuralNetwork:
    """
    neuralNetwork class
    ===================

    Attributes
    ----------
    inodes: int
        Number of nodes in the input layer.
    hnodes: int
        Number of nodes in the hidden layer.
    onodes: int
        Number of nodes in the output layer.
    lr: float
        Learning rate.
    wih: np.ndarray
        Matrix of weights from the input layer to the hidden layer.
        The initial weights are sampled from a normal distribution centered around zero and with a standard deviation 1/√(number of incoming links). They are subsequently updated during training.
    who: np.ndarray
        Matrix of weights from the hidden layer to the output layer.
        The initial weights are sampled from a normal distribution centered around zero and with a standard deviation 1/√(number of incoming links). They are subsequently updated during training.

    Methods
    -------
    activation_function:
        Sigmoid function: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html
    train:
        Update the weights for the links between the layers based on back-propagation of the error.
    query:
        Return the outputs of the neural network given an input and the weights.
    """

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        Initialize the neural network.

        Parameters
        ----------
        inputnodes: int
            Number of nodes in the input layer.
        hiddennodes: int
            Number of nodes in the hidden layer.
        outputnodes: int
            Number of nodes in the output layer.
        learningrate: float
            Learning rate.
        """
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(
            0.0,
            pow(self.hnodes, -0.5),
            (self.hnodes, self.inodes),
        )
        self.who = np.random.normal(
            0.0,
            pow(self.onodes, -0.5),
            (self.onodes, self.hnodes),
        )

        self.lr = learningrate


    @staticmethod
    def activation_function(x):
        return scipy.special.expit(x)


    def train(self, inputs_list, targets_list):
        """
        Train the neural network; update the weights for the links between the layers based on back-propagation of the error.

        Parameters
        ----------
        inputs_list: list
            List of inputs.
        targets_list: list
            List of outputs.

        Returns
        -------
        None
        """
        # convert inputs and targets to 2d arrays
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors,
        # split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            np.transpose(hidden_outputs)
        )

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs)
        )

        return None


    def query(self, inputs_list):
        """
        Query the neural network; return the outputs of the neural network given an input and the weights.

        Parameters
        ----------
        inputs_list: list
            List of inputs.

        Returns
        -------
        final_outputs: np.ndarray
        """
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
