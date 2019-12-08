import numpy as np
import os


class DNN():
    def __init__(self, input, output, layers, dropout):
        self.weights = []
        self.bias = []
        self.dropout = dropout
        self.input = input
        self.output = output
        self.layers = layers

        # weights
        for layer_num in range(len(layers)):
            if layer_num == 0:
                self.weights.append(np.random.randn(2*self.input, layers[layer_num])
                                    * np.sqrt(2 / 2*input))
            else:
                self.weights.append(np.random.randn(layers[layer_num - 1], layers[layer_num])
                                    * np.sqrt(2 / layers[layer_num - 1]))
        self.weights.append(np.random.randn(layers[-1], output)
                            * np.sqrt(2 / layers[-1]))

        # bias
        for layer_num in range(len(layers)):
            self.bias.append(np.random.randn(layers[layer_num]))
        self.bias.append(np.random.randn(output))

    def forward(self, X, train=False):
        noise = np.random.normal(0, 1, self.input)
        X = np.concatenate((X, noise))
        layer_out_before_activ = []
        layer_out_after_activ = []
        layer_out_before_activ.append(np.dot(X, self.weights[0]) + self.bias[0])
        layer_out_after_activ.append(self.sigma(layer_out_before_activ[0]))
        for weight_num in range(1, len(self.weights)):
            weights = self.weights[weight_num]
            biases = self.bias[weight_num]
            if train:
                weights = weights * np.random.binomial(1, self.dropout, size=weights.shape)
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
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def sigmaPrime(self, x):
        return 1 - np.power(self.sigma(x), 2)

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

    # def backward(self, X, y, o, model_layer, layer_out_before_activ, layer_out_after_activ):
    #     bias_improvments = []
    #     improvments = []
    #     o_error = o - y
    #     o_delta = cal_delta(X, model_layer, self.networkData["mathFun"], self.networkData["dataOutputDim"], self.networkData["mathFunName"])
    #     o_delta = o_error * o_delta
    #     last_improvments = []
    #     for i in range(self.networkData["imannParamNum"]):
    #         last_improvments.append(layer_out_after_activ[-1][:, i].dot(o_delta[i]))
    #
    #     # l-1 layer update
    #     last_deltas = []
    #     for i in range(self.networkData["imannParamNum"]):
    #         last_deltas.append(o_delta[i].dot(self.networkData["weights"][-1][i]).reshape((-1, 1)))
    #     last_deltas = np.array(last_deltas).reshape((-1, self.networkData["imannParamNum"]))
    #     a = self.bentPrime(layer_out_before_activ[-1])
    #     delta = last_deltas * a
    #     bias_improvments.append(np.sum(delta,axis=0))
    #     improvments.append(layer_out_after_activ[-2].T.dot(delta))
    #
    #     # other layer update
    #     for layer_num in range(-2, -len(self.networkData["weights"]), -1):
    #         if layer_num == -len(self.networkData["weights"])+1:
    #             delta = delta.dot(self.networkData["weights"][layer_num].T)
    #             delta = delta * self.bentPrime(layer_out_before_activ[layer_num])
    #             bias_improvments.append(np.sum(delta, axis=0))
    #             improvments.append(X.T.dot(delta))
    #         else:
    #             delta = delta.dot(self.networkData["weights"][layer_num].T)
    #             delta = delta * self.bentPrime(layer_out_before_activ[layer_num])
    #             bias_improvments.append(np.sum(delta, axis=0))
    #             improvments.append(layer_out_after_activ[layer_num - 1].T.dot(delta))
    #     num = -2
    #     for improvment in improvments:
    #         self.networkData["weights"][num] -= self.networkData["bpLearningRate"] * improvment
    #         num -= 1
    #     for i in range(self.networkData["imannParamNum"]):
    #         self.networkData["weights"][-1][i] -= self.networkData["bpLearningRate"] * last_improvments[i]
    #     num = -1
    #     for improvment in bias_improvments:
    #         self.networkData["biases"][num] += self.networkData["bpLearningRate"] * improvment
    #         num -= 1
    #
    # def train(self, epochs=100):
    #     x = self.networkData["trainX"]
    #     y = self.networkData["trainY"]
    #     for i in range(epochs):
    #         o, model_layer, layer_out_before_activ, layer_out_after_activ = self.forward(X=x, train=True)
    #         self.backward(x, y, o, model_layer, layer_out_before_activ, layer_out_after_activ)
    #
    def load_network(self):
        networkData = np.load("dnn.npy", allow_pickle=True).item()
        self.weights = networkData["weights"]
        self.bias = networkData["biases"]

    def save_network(self):
        np.save("dnn.npy", {"weights": self.weights, "biases": self.bias})
