import numpy as np


class Neural_Network:
    """ Intialized Parameters """

    def __init__(self, layer_dims, Y, learning_rate):  # layers_dims is the dimensions of each layer
        self.L = len(layer_dims)  # number of layers in NN
        self.Network = {}
        self.Y = Y
        self.learning_rate = learning_rate
        for l in range(1, self.L):
            self.Network['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
            self.Network['b' + str(l)] = np.zeros((layer_dims[l], 1))

    """ Activation Function """

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid_backward(self, dA, Z):
        s = self.sigmoid(Z)
        dZ = dA * s * (1 - s)
        return dZ

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    """ Cost Calculation """

    def Cost(self, AL, Y):
        m = Y.shape[1]
        cost = (1.0 / m) * (- np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
        return np.squeeze(cost)

    """ Train the Neural Network """

    def Train_NN(self, l, A_prev):
        W = self.Network['W' + str(l)]
        b = self.Network['b' + str(l)]
        Z = W.dot(A_prev) + b  # Z = W * A[l-1] + b

        if l == self.L - 1:  # last layer mean to output layer for both forward and backword prop
            A = self.sigmoid(Z)
            Y = self.Y
            dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
            dZ = self.sigmoid_backward(dA, Z)
        else:  # other L-1 Layers for both forward and backward prop
            A = self.relu(Z)
            dA = self.Train_NN(l + 1,
                               A)  # Here dA comes from our recursive call dA_prev that we return from our last recursive call
            dZ = self.relu_backward(dA, Z)
        m = A_prev.shape[1]
        dW = 1.0 / m * np.dot(dZ, A_prev.T)
        db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        self.Network['W' + str(l)] -= self.learning_rate * dW
        self.Network['b' + str(l)] -= self.learning_rate * db

        return dA_prev  # we return our first dA_prev here then further

    """ Test_NN """

    def Test_NN(self, X, Y, print_cost=False):
        m = X.shape[1]
        p = np.zeros((1, m))
        A_prev = X
        for l in range(1, self.L - 1):
            Z = self.Network['W' + str(l)].dot(A_prev) + self.Network['b' + str(l)]
            A_prev = self.relu(Z)

        Z = self.Network['W' + str(self.L - 1)].dot(A_prev) + self.Network['b' + str(self.L - 1)]
        Output = self.sigmoid(Z)
        if print_cost:   costs = self.Cost(Output, Y)

        for i in range(0, Output.shape[1]):
            if Output[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        Accuracy = np.sum((p == Y) / m)
        if print_cost: return p, costs, Accuracy
        return p, Accuracy