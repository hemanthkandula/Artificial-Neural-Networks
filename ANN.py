import os
from itertools import groupby, cycle

import numpy as np
import pandas as pd


def sigmoid_activation(z, derivative=False):
    if derivative:
        return sigmoid_activation(z) * (1 - sigmoid_activation(z))
    else:
        return 1 / (1 + np.exp(-z))


def relu(x):

    return np.maximum(0,x)
def relu_deriv(deltaX,Y):
    deltaY = np.array(deltaX,copy=True)
    deltaY[Y<=0] = 0
    return deltaY

def sigmoid_backward(dA, Z):
    sig = sigmoid_activation(Z)
    return dA * sig * (1 - sig)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;
def softmax(x):
    """from https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"""
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class FullyConnectedLayer:
    def __init__(self, number_of_neurons, number_of_neurons_prev_layer, output_layer=False, weights=None, biases=None):

        self.output_layer = output_layer
        self.number_of_neurons = number_of_neurons

        if weights is None and biases is None:
            self._weights = np.random.rand(number_of_neurons, number_of_neurons_prev_layer)
            self._biases = np.random.rand(number_of_neurons)

        else:
            self._weights = weights
            self._biases = biases

    def output(self, inputs):
        # print(self._weights.shape, inputs.shape,self._biases.shape)
        # + np.dot(self._weights, inputs)
        output_before_activation = np.dot(self._weights, inputs) + self._biases

        if self.output_layer:
            # layer_output = softmax(output_before_activation)
            layer_output = sigmoid_activation(output_before_activation)
        else:
            layer_output = relu(output_before_activation)

        self.outputs = layer_output
        self.outputs_derv = relu(output_before_activation)

        self.output_before_activation = output_before_activation
        self.inputs = inputs

        return output_before_activation, layer_output



    def backward(self,delta_curr):

        prev_layer_len = self.inputs.shape[1]

        if self.output_layer:

            deltaBA = sigmoid_backward(delta_curr,self.outputs)

            # derivative of the matrix W
            dW_curr = np.dot(dZ_curr, A_prev.T) / m
            # derivative of the vector b
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
            # derivative of the matrix A_prev
            dA_prev = np.dot(W_curr.T, dZ_curr)




    def __str__(self):
        return "FullyConnectedLayer with " + str(self.number_of_neurons) \
               + ": weights shape = " + str(self._weights.shape) + "  biases shape = " + str(self._biases.shape)


class NeuralNetwork:
    def __init__(self, inputs, hidden_layers, output):
        self.hidden_layers = hidden_layers
        self.inputs = inputs
        self.output = output
        self.layers = []

    # def get_weights_bias(self):

    def __str__(self):
        stri = ""
        for layer in self.layers:
            stri + layer.__str__() + "\n"
            print(layer)

        return stri
        # stri = "Fully connected layer with  "+ num_neurons""

    def built(self):
        self.layers.append(FullyConnectedLayer(self.hidden_layers[0], self.inputs))  # First layer

        for prev_layer_neurons, number_of_neurons in zip(self.hidden_layers, self.hidden_layers[1:]):
            self.layers.append(FullyConnectedLayer(number_of_neurons, prev_layer_neurons))  # hidden layers

        self.layers.append(FullyConnectedLayer(self.output, self.hidden_layers[-1], output_layer=True))  # output layer

    def forward_propagation(self, x):
        for layer in self.layers:
            before_activation, x = layer.output(x)

        return before_activation, x

    def back_propagation(self, train_y):

        # ouptut_error =
        deltas = []
        deltas.append((train_y - self.layers[-1].outputs) * self.layers[-1].outputs_derv)

        for layer in reversed(self.layers[:-1]):

            deltas.append(layer.outputs_derv * np.dot(layer.weights,deltas[-1]))

            layer.weights += learning_rate * layer.outputs*deltas[-1]





        pass

    def train(self, num_iterations, train_data, train_label, validation_data, validation_label):
        train_data_iterator, train_label_iterator = \
            cycle(train_data), cycle(train_label)

        weight = np.array([])
        biases = np.array([])

        valid_mse = []
        current_mse = []

        for itr in range(num_iterations):
            train_x, train_y = next(train_data_iterator), next(train_label_iterator)

            self.forward_propagation(train_x)

            desired_output = one_hot_encoding(train_y)

            self.back_propagation(desired_output)



            valid_mse_cur =[]

            for x, y in zip(validation_data, validation_label):
                _, actual_output = self.forward_propagation(x)

                desired_output = one_hot_encoding(y)
                mse = self.mse(desired_output, actual_output)
                valid_mse_cur.append(mse)

            valid_mse = sum(valid_mse_cur)

            valid_mse_cur.append(valid_mse)



            # if len(current_mse) < 5:
            #     current_mse.append(valid_mse)
            # else:
            #     if valid_mse > np.max(current_mse):
            #         print(" STOP training")



    def back_prop2(self,y_train):
        pass


    def mse(self, d, o):

        return np.sum((d - o) ** 2) / len(d)

    def test(self, test_data, test_label):
        pass

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_
def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()




def one_hot_encoding(x):
    label = np.zeros(len(classes))
    label[(x - 1)] = 1

    return label


if __name__ == '__main__':
    # dataset_pd = pd.read_csv('ANN - Iris data.txt', header=None,index_col=None )
    # dataset_pd.columns = ['S_L', 'S_W', 'P_L', 'P_W' ,'classes']
    # print(dataset_pd.columns)
    #
    # dataset_pd.groupby('classes')
    #
    # df_class_1 = dataset_pd.loc[dataset_pd['classes'] == "Iris-setosa"]

    data_file = open("ANN - Iris data.txt", "r")
    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    data_list = data_file.readlines()
    X = []
    Y = []
    data = []
    for data_point in data_list:
        data_point_list = data_point.strip().split(",")
        data_point_list[4] = str(classes.index(data_point_list[4]))

        data.append(data_point_list)

    # data = np.array(data)
    train = []
    test = []
    validation = []

    for k, class_group in groupby(data, lambda x: x[4]):
        class_group = list(class_group)
        train.extend(class_group[int(len(class_group) * 0): int(len(class_group) * 0.6)])
        validation.extend(class_group[int(len(class_group) * 0.6): int(len(class_group) * 0.8)])
        test.extend(class_group[int(len(class_group) * 0.8): int(len(class_group) * 1.0)])

    train = np.array(train, dtype=float)
    validation = np.array(validation, dtype=float)
    test = np.array(test, dtype=float)

    x_train = train[:, :-1]
    x_validation = validation[:, :-1]
    x_test = test[:, :-1]

    y_train = train[:, -1]
    y_validation = validation[:, -1]
    y_test = test[:, -1]

    # train validation test split

    input_features = 4
    output_feature = 3

    learning_rate = 0.1


    neural_network = NeuralNetwork(inputs=input_features, hidden_layers=[5, 5, 5], output=output_feature)
    neural_network.built()

    number_of_iterations = 1000

    print(neural_network.forward_propagation(x_train[0]))

    print(neural_network)
    neural_network.train(num_iterations=number_of_iterations,
                         train_data=x_train, train_label=y_train,
                         validation_data=x_validation, validation_label=y_validation)
    #
    # neural_network.test(
    #     test_data=x_train, test_label=y_train,
    # )
