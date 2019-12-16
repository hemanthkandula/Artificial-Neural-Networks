import os
from itertools import groupby

import numpy as np
import pandas as pd


def sigmoid_activation(z, derivative=False):
    if derivative:
        return sigmoid_activation(z) * (1 - sigmoid_activation(z))
    else:
        return 1 / (1 + np.exp(-z))


def relu(x):
    return x * (x >= 0.)


def softmax(x):
    """from https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"""
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class FullyConnectedLayer:
    def __init__(self, number_of_neurons, number_of_neurons_prev_layer):



        self._weights = np.random.rand( number_of_neurons,number_of_neurons_prev_layer)
        self._biases = np.random.rand(number_of_neurons)

    def output(self, inputs, output_layer=False):
        # print(self._weights.shape, inputs.shape,self._biases.shape)
        # + np.dot(self._weights, inputs)
        output_before_activation = np.dot(self._weights ,inputs)  + self._biases


        if output_layer:
            layer_output = softmax(output_before_activation)
        else:
            layer_output = relu(output_before_activation)

        return output_before_activation, layer_output


class NeuralNetwork:
    def __init__(self, inputs, hidden_layers, output):
        self.hidden_layers = hidden_layers
        self.inputs = inputs
        self.output = output
        self.layers = []




    def built(self):
        self.layers.append(FullyConnectedLayer(self.hidden_layers[0], self.inputs))  # input layer

        for prev_layer_neurons, number_of_neurons in zip(self.hidden_layers, self.hidden_layers[1:]):
            self.layers.append(FullyConnectedLayer( number_of_neurons,prev_layer_neurons))  # hidden layers

        self.layers.append(FullyConnectedLayer( self.output,self.hidden_layers[-1]))  # output layer

    def forward_propagation(self, x):
        for layer in self.layers:
            before_activation, x = layer.output(x)

        return before_activation, x

    def back_propagation(self):
        #         TODO :

        pass

    def train(self,num_iterations, train_data,train_label,validation_data,validation_label):
        pass

    def test(self, test_data, test_label):
        pass


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






    x_train = train[:,:-1]
    x_validation = validation[:,:-1]
    x_test = test[:,:-1]

    y_train = train[:,-1]
    y_validation = validation[:,-1]
    y_test = test[:,-1]


    # train validation test split

    input_features = 4
    output_feature = 1

    neural_network = NeuralNetwork(inputs=4, hidden_layers=[5, 5, 5], output=3)
    neural_network.built()

    number_of_iterations = 1000


    print(neural_network.forward_propagation(x_train[0]))


    neural_network.train(num_iterations=number_of_iterations,
                         train_data=x_train,train_label=y_train,
                         validation_data=x_validation,validation_label=y_validation)

    neural_network.test(
                         test_data=x_train,test_label=y_train,
                         )