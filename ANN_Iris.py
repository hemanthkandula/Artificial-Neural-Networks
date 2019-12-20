# Names:	Nathaniel Mahlum 	(Nathaniel.Mahlum@tufts.edu)
#           Sai Hemanth Kandula (Sai_Hemanth.Kandula@tufts.edu)
# Assignment 5: Artificial Neural Networks
# __________________________________________________________________


from itertools import groupby

import numpy as np

import matplotlib.pyplot as plt

seed = 15

np.random.seed(seed=seed)


# Sigmoid activation function that calculates the
# output of the hidden layers

def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))


# Returns the derivative of each sigmoid value

def sigmoid_derivative(s):
    return s * (1 - s)


# Returns 0 if the inputted value is negative

def relu(x):
    return np.maximum(0, x)


# Takes in a array of values and normalizes it into
# a probability distribution using the softmax method

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))

    return exps / np.sum(exps, axis=1, keepdims=True)


# Takes a value in the range of [0,2] and
# returns the one-hot-encoding value
# for the value passed in

def one_hot_encoding(x):
    label = np.zeros(len(classes))

    label[(int(x) - 1)] = 1

    return label


class NeuralNetwork:

    # The build function for the Neural Network takes in the number of inputs
    # the number of outputs, the number of desired neurons, and the learning
    # rate value, and initializes a Neural Network with one hidden layer
    # the number of neurons passed in

    def build(self, input_dim, output_dim, num_neurons, learning_rate):

        self.lr = learning_rate

        self.input_dim = input_dim

        self.output_dim = output_dim

        self.num_neurons = num_neurons

        # Initialize weights and biases connecting each layer

        self.w1 = np.random.randn(input_dim, num_neurons)

        self.b1 = np.zeros((1, num_neurons))

        self.w2 = np.random.randn(num_neurons, num_neurons)

        self.b2 = np.zeros((1, num_neurons))

        self.w3 = np.random.randn(num_neurons, output_dim)

        self.b3 = np.zeros((1, output_dim))

    # Calculates the ouputs error

    def output_error(self, pred, real):
        n_samples = real.shape[0]
        res = pred - real
        return res / n_samples


    # Calculates the mean squared error
    @staticmethod
    def MSE(yHat, y):
        return np.sum((yHat - y) ** 2) / y.size

    # feedforward performs forward propagation, building out an instance of the
    # ANN using inputs that are passed-in, calling the activation function on each
    # calculated output and then progressing through the hidden layer
    # to calculate the output using the softmax method

    def feedforward(self, data):

        before_activation_1 = np.dot(data, self.w1) + self.b1

        self.layer_1_output = sigmoid_activation(before_activation_1)

        before_activation_2 = np.dot(self.layer_1_output, self.w2) + self.b2

        self.layer_2_output = sigmoid_activation(before_activation_2)

        before_activation_3 = np.dot(self.layer_2_output, self.w3) + self.b3

        self.output_layer_output = softmax(before_activation_3)

    # Performs backwards propagation by going through and calculating
    # the output error present in each layer, and then updating the weights
    # and biases accordingly

    def backprop(self):

        activation_3_error = self.output_error(self.output_layer_output, self.train_label)

        before_activation_L2_error = np.dot(activation_3_error, self.w3.T)

        L2_output_error = before_activation_L2_error * sigmoid_derivative(self.layer_2_output)

        before_activation_L1_error = np.dot(L2_output_error, self.w2.T)

        L1_output_error = before_activation_L1_error * sigmoid_derivative(self.layer_1_output)

        self.w3 -= self.lr * np.dot(self.layer_2_output.T, activation_3_error)

        self.b3 -= self.lr * np.sum(activation_3_error, axis=0, keepdims=True)

        self.w2 -= self.lr * np.dot(self.layer_1_output.T, L2_output_error)

        self.b2 -= self.lr * np.sum(L2_output_error, axis=0)

        self.w1 -= self.lr * np.dot(self.train_data.T, L1_output_error)

        self.b1 -= self.lr * np.sum(L1_output_error, axis=0)

    # Feeds data into the algorithm, takes the projected outputs,
    # calculates the logistic loss and then evaluates the accuracy of the data,
    # returning the total loss of the algorithm as well as the testing accuracy

    def evaluate(self, data, labels):

        self.feedforward(data)

        preds = self.output_layer_output.argmax(axis=1)

        log_loss = self.MSE(self.output_layer_output, labels)

        acc = np.mean(preds == labels.argmax(axis=1))

        return log_loss, acc

    # Handles the training part of the algorithm, taking in the number of desired iterations
    # the training data, the training labels, as well as the validation data and labels
    # in order to test the increasing accuracy of the algorithm

    def train(self, num_of_iterations, train_data, train_label, valid_data, valid_label):

        # Initializes data structures
        training_accs = []

        training_losses = []

        validation_accs = []

        validation_losses = []

        self.train_data = train_data

        self.train_label = train_label

        self.valid_data = valid_data

        self.valid_label = valid_label

        print()

        print("Training started")

        print("_" * 125)

        # Iterates back propagation

        for itr in range(num_of_iterations):

            # Feeds the training data into the algorithm
            # and then performs back propagation

            self.feedforward(self.train_data)
            self.backprop()

            # Calculates the training loss and accuracy and the validation
            # loss and accuracy, appending these values to the structures storing
            # accuracy and loss values

            train_loss, train_acc = self.evaluate(self.train_data, self.train_label)

            valid_loss, valid_acc = self.evaluate(self.valid_data, self.valid_label)

            training_accs.append(train_acc)

            training_losses.append(train_loss)

            validation_accs.append(valid_acc)

            validation_losses.append(valid_loss)

            # Prints out results every 50 iterations

            if itr % 50 == 0:
                print("Iteration: {:05} - loss: {:.5f} - training accuracy: {:.5f} % | validation loss:"

                      " {:.5f} - validation accuracy: {:.5f} %".format(itr, train_loss, train_acc * 100, valid_loss,
                                                                       valid_acc * 100))

        # Plots out training loss and validation loss on a curve, outputting
        # loss curves image

        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.plot(training_losses, label='training_loss')

        ax.plot(validation_losses, label='validation_loss')

        ax.legend()

        fig.savefig("Loss_curves.png")

        plt.close(fig)

        # Plots out training acc and validation acc on a curve, outputting
        # acc curves image
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.plot(training_accs, label='training accuracy')

        ax.plot(validation_accs, label='validation accuracy')

        ax.legend()

        fig.savefig("Accuracy_curves.png")

        plt.close(fig)

    # Testing function that allows the user to call this on
    # the testing data and receives the accuracy and loss value
    # for the data passed in after the ANN has been trained

    def test(self, test_data, test_labels):

        return self.evaluate(test_data, test_labels)


if __name__ == '__main__':

    # Reads in the iris data

    data_file = open("ANN - Iris data.txt", "r")

    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    data_list = data_file.readlines()

    # Initializes data structure and then reads it in in the
    # desired format

    data = []

    for data_point in data_list:
        data_point_list = data_point.strip().split(",")

        data_point_list[4] = str(classes.index(data_point_list[4]) + 1)

        data.append(data_point_list)

    # Initializes training, testing and validation portions

    train = []

    test = []

    validation = []

    # Scrambles the data randomly and puts the correct proportions of data into
    # each category of data (

    for k, class_group in groupby(data, lambda x: x[4]):
        class_group = list(class_group)

        np.random.shuffle(class_group)

        train.extend(class_group[int(len(class_group) * 0): int(len(class_group) * 0.5)])

        validation.extend(class_group[int(len(class_group) * 0.5): int(len(class_group) * 0.75)])

        test.extend(class_group[int(len(class_group) * 0.75): int(len(class_group) * 1.0)])

    # Converts values to floats to be used in the algorithm

    train = np.array(train, dtype=float)

    validation = np.array(validation, dtype=float)

    test = np.array(test, dtype=float)

    # Seperates data into input data and output labels

    x_train = train[:, :-1]

    x_validation = validation[:, :-1]

    x_test = test[:, :-1]

    y_train = train[:, -1]

    y_validation = validation[:, -1]

    y_test = test[:, -1]

    # Convert all labels into one-hot encoding form

    y_test = np.array([one_hot_encoding(k) for k in y_test])

    y_validation = np.array([one_hot_encoding(k) for k in y_validation])

    y_train = np.array([one_hot_encoding(k) for k in y_train])

    print()

    # Prints out dimensions of training, validation and testing data

    print("Training data shape ", x_train.shape, y_train.shape)

    print("Validation data shape", x_validation.shape, y_validation.shape)

    print("Test data shape", x_test.shape, y_test.shape)

    print()

    # Prints out how labels are encoded

    for class_name in classes:
        print("Class:", class_name, ": encoded as -", one_hot_encoding(classes.index(class_name) + 1))

    print()

    neuralnetwork = NeuralNetwork()

    # Builds the neural network

    neuralnetwork.build(input_dim=x_train.shape[1], output_dim=y_train.shape[1], num_neurons=50, learning_rate=0.01)

    # Trains the network
    neuralnetwork.train(num_of_iterations=2000, train_data=x_train, train_label=y_train, valid_data=x_validation,
                        valid_label=y_validation)

    # Test network using the testing data set in place of the gardener input
    test_loss, test_acc = neuralnetwork.test(test_data=x_test, test_labels=y_test)

    # Prints out results from user testing (from test data)

    print("_" * 125)

    print("Test data Accuracy : {:.5f} % | loss: {:.5f}  ".format(test_acc * 100, test_loss))

    print("_" * 125)
