import  os
import numpy as np




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
    def __init__(self, number_of_neurons_prev_layer,number_of_neurons):

        self._weights= np.random.rand(number_of_neurons_prev_layer, number_of_neurons)
        self._biases = [np.random.rand(1, number_of_neurons)]



    def output(self,inputs,output_layer=False):

        output_before_activation = np.dot(self._weights, inputs) + self._biases
        if output_layer :
            layer_output = sigmoid_activation(output_before_activation)
        else:
            layer_output = relu(output_before_activation)

        return output_before_activation,layer_output

class NeuralNetwork:
    def __init__(self, inputs ,hidden_layers,output):
        self.hidden_layers=hidden_layers
        self.inputs = inputs
        self.output = output
        self.layers = []


    def built(self):
        self.layers.append(FullyConnectedLayer(self.hidden_layers[0],len(self.inputs) ))           # input layer

        for prev_layer_neurons,number_of_neurons in zip(self.hidden_layers, self.hidden_layers[1:]):
            self.layers.append(FullyConnectedLayer(prev_layer_neurons, number_of_neurons))         # hidden layers

        self.layers.append(FullyConnectedLayer(self.hidden_layers[-1],self.output ))               # output layer





    def forward_propagation(self,x):
        for layer in self.layers:
            before_activation, x = layer.output(x)



    def back_propagation(self):
        pass
#         TODO :





if __name__ == '__main__':

    input_features = 4
    output_feature = 1



    neural_network = NeuralNetwork()



















