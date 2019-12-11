import numpy as np
import os


class DNN():
    def __init__(self, input, output, layers):
        self.weights = []
        self.bias = []
        self.input = input
        self.output = output
        self.layers = layers
        self.draw = 0
        self.wins = 0
        self.loses = 0
        self.bad_moves = 0

        # weights
        for layer_num in range(len(layers)):
            if layer_num == 0:
                self.weights.append(np.random.randn(self.input, layers[layer_num])
                                    * np.sqrt(2 / input))
            else:
                self.weights.append(np.random.randn(layers[layer_num - 1], layers[layer_num])
                                    * np.sqrt(2 / layers[layer_num - 1]))
        self.weights.append(np.random.randn(layers[-1], output)
                            * np.sqrt(2 / layers[-1]))

        # bias
        for layer_num in range(len(layers)):
            self.bias.append(np.random.randn(layers[layer_num]))
        self.bias.append(np.random.randn(output))

    def forward(self, X):
        layer_out_before_activ = []
        layer_out_after_activ = []
        layer_out_before_activ.append(np.dot(X, self.weights[0]) + self.bias[0])
        layer_out_after_activ.append(self.bent(layer_out_before_activ[0]))
        for weight_num in range(1, len(self.weights)):
            weights = self.weights[weight_num]
            biases = self.bias[weight_num]
            layer_out_before_activ.append(np.dot(layer_out_after_activ[weight_num - 1], weights) + biases)
            layer_out_after_activ.append(self.sigma(layer_out_before_activ[weight_num]))
        o = layer_out_after_activ[-1].copy()
        return o

    def predict(self, x):
        y = self.forward(x)
        return y

    def bent(self, x):
        return ((np.sqrt(np.power(x, 2) + 1) - 1) / 2) + x

    def bentPrime(self, x):
        return x / (2 * np.sqrt(np.power(x, 2) + 1)) + 1

    def sigma(self, x):
        return 1/(1 + np.exp(-x))

    def sigmaPrime(self, x):
        return self.sigma(x) * (1-self.sigma(x))

    def toVector(self):
        V = np.array([])
        for w in self.weights:
            V = np.append(V, w.reshape(-1, np.size(w))[0])
        for b in self.bias:
            V = np.append(V, b.reshape(-1, np.size(b))[0])
        return V

    def fromVector(self, V):
        i = 0
        for idx, w in enumerate(self.weights):
            k = np.size(w)+i
            self.weights[idx] = V[i:k].reshape(w.shape)
            i = k
        for idx, b in enumerate(self.bias):
            k = np.size(b)+i
            self.bias[idx] = V[i:k].reshape(b.shape)
            i = k

    def load_network(self):
        networkData = np.load("dnn.npy", allow_pickle=True).item()
        self.weights = networkData["weights"]
        self.bias = networkData["biases"]

    def save_network(self):
        np.save("dnn.npy", {"weights": self.weights, "biases": self.bias})

    def reset_fitness(self):
        self.draw = 0
        self.wins = 0
        self.loses = 0
        self.bad_moves = 0
