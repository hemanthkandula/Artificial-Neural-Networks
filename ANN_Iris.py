from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt

seed = 15

np.random.seed(seed=seed)
def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(s):
    return s * (1 - s)
def relu(x):
    return np.maximum(0,x)


def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)


# def softmax(x):
#     """from https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"""
#     """Compute softmax values for each sets of scores in x."""
#     return np.exp(x) / np.sum(np.exp(x), axis=0)



def one_hot_encoding(x):   #
    label = np.zeros(len(classes))

    label[(int(x) - 1)] = 1

    return label



class NeuralNetwork:
    def build(self, input_dim,output_dim,num_neurons,learning_rate ):
        self.lr = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons


        self.w1 = np.random.randn(input_dim, num_neurons)
        self.b1 = np.zeros((1, num_neurons))
        self.w2 = np.random.randn(num_neurons, num_neurons)
        self.b2 = np.zeros((1, num_neurons))
        self.w3 = np.random.randn(num_neurons, output_dim)
        self.b3 = np.zeros((1, output_dim))



    def cross_entropy_loss(self,pred, real):
        n_samples = real.shape[0]
        res = pred - real
        return res / n_samples


    @staticmethod
    def logistic_loss(pred, real):
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        loss = np.sum(logp) / n_samples
        return loss

    def feedforward(self, data):
        before_activation_1 = np.dot(data, self.w1) + self.b1
        self.layer_1_output = sigmoid_activation(before_activation_1)
        before_activation_2 = np.dot(self.layer_1_output, self.w2) + self.b2
        self.layer_2_output = sigmoid_activation(before_activation_2)
        before_activation_3 = np.dot(self.layer_2_output, self.w3) + self.b3
        self.output_layer_output = softmax(before_activation_3)

    def backprop(self):
        activation_3_error = self.cross_entropy_loss(self.output_layer_output, self.train_label)
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

    def predict(self, data):
        self.feedforward(data)
        probs = np.array([np.argmax(k) + 1 for k in self.output_layer_output])
        return probs

    def evaluvate(self,data,labels):
        self.feedforward(data)
        # preds = np.array([np.argmax(k) + 1 for k in self.output_layer_output])
        preds = self.output_layer_output.argmax(axis=1)
        log_loss = self.logistic_loss(self.output_layer_output, labels)
        acc = np.mean(preds == labels.argmax(axis=1))
        return log_loss,acc


    def train(self,num_of_iterations,train_data,train_label,valid_data,valid_label):
        training_accs= []
        training_losses = []
        validation_accs = []
        validation_losses = []

        self.train_data =train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        print()
        print("Training started")
        print("_" * 125)



        for itr in range(num_of_iterations):
            self.feedforward(self.train_data)
            self.backprop()
            # print(nn_.predict(x_train))
            # probs = np.array([np.argmax(k) + 1 for k in nn_.output_layer_output])
            train_loss, train_acc = self.evaluvate(self.train_data, self.train_label)
            valid_loss, valid_acc = self.evaluvate(self.valid_data,self.valid_label)
            training_accs.append(train_acc)
            training_losses.append(train_loss)
            validation_accs.append(validation_accs)
            validation_losses.append(valid_loss)

            if itr % 50==0:
                print("Iteration: {:05} - loss: {:.5f} - training accuracy: {:.5f} % | validation loss:"
                      " {:.5f} - validation accuracy: {:.5f} %".format(itr, train_loss, train_acc*100,valid_loss,valid_acc*100))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(training_losses, label='training_loss')
        ax.plot(validation_losses, label='validation_loss')
        ax.legend()

        fig.savefig("Loss_curves.png")
        # plt.show()
        plt.close(fig)
        #
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.plot(training_accs, label='training accuracy')
        # ax.plot(validation_accs, label='validation accuracy')
        #
        # ax.legend()
        # fig.savefig("Accuracy_curves.png")
        # plt.show()
        #
        # plt.close(fig)
        # plt.show()


    def test(self,test_data,test_labels):
        return self.evaluvate(test_data,test_labels)



if __name__ == '__main__':
    data_file = open("ANN - Iris data.txt", "r")
    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    data_list = data_file.readlines()
    X = []
    Y = []
    data = []
    for data_point in data_list:
        data_point_list = data_point.strip().split(",")
        data_point_list[4] = str(classes.index(data_point_list[4])+1)

        data.append(data_point_list)

    # data = np.array(data)
    train = []
    test = []
    validation = []
    # train validation test split
    # Train set 50%
    # validation set 25 %
    # test set 25 %
    for k, class_group in groupby(data, lambda x: x[4]):
        class_group = list(class_group)
        np.random.shuffle(class_group)
        train.extend(class_group[int(len(class_group) * 0): int(len(class_group) * 0.5)])
        validation.extend(class_group[int(len(class_group) * 0.5): int(len(class_group) * 0.75)])
        test.extend(class_group[int(len(class_group) * 0.75): int(len(class_group) * 1.0)])

    train = np.array(train, dtype=float)
    validation = np.array(validation, dtype=float)
    test = np.array(test, dtype=float)

    x_train = train[:, :-1]
    x_validation = validation[:, :-1]
    x_test = test[:, :-1]

    y_train = train[:, -1]
    y_validation = validation[:, -1]
    y_test = test[:, -1]


    y_test = np.array([one_hot_encoding(k) for k in y_test])
    y_validation = np.array([one_hot_encoding(k) for k in y_validation])
    y_train = np.array([one_hot_encoding(k) for k in y_train])
    print()

    print("Training data shape ",x_train.shape,y_train.shape)
    print("Validation data shape",x_validation.shape,y_validation.shape)
    print("Test data shape", x_test.shape,y_test.shape)
    print()
    for class_name in classes:
        print("Class:", class_name,": encoded as -",one_hot_encoding(classes.index(class_name)+1))
    print()
    neuralnetwork = NeuralNetwork()
    # probs_real_train = np.array([np.argmax(k) + 1 for k in y_train])
    # probs_real_valid = np.array([np.argmax(k) + 1 for k in y_validation])
    neuralnetwork.build(input_dim=x_train.shape[1],output_dim=y_train.shape[1],num_neurons=50,learning_rate=0.01)
    neuralnetwork.train(num_of_iterations=2000,train_data=x_train, train_label=y_train, valid_data=x_validation, valid_label=y_validation)
    test_loss,test_acc = neuralnetwork.test(test_data=x_test,test_labels=y_test)
    print("_"*125)
    print("Test data Accuracy : {:.5f} % | loss: {:.5f}  ".format(test_acc*100,test_loss))
    print("_"*125)





    # for itr in range(1000):
    #     nn_.feedforward()
    #     nn_.backprop()
    #     # print(nn_.predict(x_train))
    #     probs = np.array([np.argmax(k)+1 for k in nn_.a3])
    #
    #     print(accuracy_score(probs_real_train,probs))
    #
    #
    # preds = nn_.predict(x_validation)
    # print(accuracy_score(probs_real_valid,preds))
    #
    #
    #



