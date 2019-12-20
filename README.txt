Names:	Nathaniel Mahlum 	(Nathaniel.Mahlum@tufts.edu)
        Sai Kandula    		(Sai_Hemanth.Kandula@tufts.edu)
Assignment 5: Artificial Neural Networks
______________________________________________________________________
File Specifications

Files:
README
ANN_Iris.py
Accuracy_curves.png (generated file)
Loss_curves.png (generated file)

ANN_Iris.py:
    
    This file contains our implementation of an Artificial Neural Network
    in constructing an algorithm that identifies a type of iris given four
    key floral dimensions at high accuracy. It does this through four external
    functions, an NeuralNetwork class, and a main function that calls internal
    functions on a NeuralNetwork instance.
    
    In the following, we will outline key functions used, the NeuralNetwork
    class and its strategy, as well as how it all fits together and how
    the NeuralNetwork can be tested by the user.
    
        1. sigmoid_activation(z) is the activation function used in the NeuralNetwork
        to calculate the output of each neuron in the hidden layer
    
        2. sigmoid_derivative(s) returns the derivative of each sigmoid value to be used
        in the back-propagation portion of the algorithm
        
        3. relu(x) handles negative values by returning 0 if the inputted value
        is negative
        
        4. softmax(s) takes a distribution and returns the probabilistic distribution
        through normalization using the softmax method
        
        5. one_hot_encoding(x) is used to turn values that store the classification
        of the type of iris into a one-hot encoding method.
        
        6. class NeuralNetwork stores the structure of a Neural Network with a single
        hidden layer. The network is structured such that the number of input and output
        neurons is dynamic and is determined by user input
        
            a. build: The build function takes in the desired number of input and output
            neurons, as well as the size of the hidden layer, and constructs a 
            neuralnetwork of those dimensions, randomly generating weights and biases
            between those internal neurons
            
            b. MSE loss and output error: MSE calculates the mean squared error values
            to be used when validating back-propagation. When validation loss begins to
            increase, the ANN should not be trained more as error is likely to increase
            due to over fitting.The validation loss  allows us to identify this.
            output_error function is used to find the error with neuralnetwork predicted
            output and real (labelled) output
            
            c. feedforward: performs forward propagation by passing in the input values
            through the algorithm and calling sigmoid activation in order to get the
            output value of each hidden neuron as well as the output neurons. On the
            output layer, the function calls the aforementioned softmax method to
            normalize output to a probability distribution
            
            d. backprop: performs back-propagation by calculating the delta values of
            each neuron and adjusting weights and biases respectively. 
            
            e. evaluate: takes in structures storing input and output data and runs
            it through the algorithm, calculating the accuracy and loss (using the
            MSE loss function discussed earlier) when it is ran through
            before returning it. This is used in assessing the accuracy of the
            Neural Network in validating and testing
            
            f. train: handles the training of the Neural Network by handling inputted
            training and validation data. In each iteration, it feeds in the training
            data and runs back propagation on this before evaluating the data on both
            the training and the validation data, updating accuracy and loss values.
            After iterating through, it generates a plot representing the mean squared
            error loss values and another plot for accuracies values for training set
            and validation set on each iteration, outputted as "Loss_curves.png" and
            "Accuracy_curves.png"
            
            g. test: allows the user to call this function by inputting testing data
            and correct classifications as parameters, receiving accuracy and loss values
            on these inputs after running the inputs through the neural network and
            receiving the predicted outputs from the ANN.
            
        7. main(): Handles the data by loading it into a datastructure and then seperating
        it into three different categories of training (50%), validation (25%) and
        testing (25%) data. The test data, as stated on the spec, is used in place of
        user / gardener input. Each section of data is stored in two seperate structures,
        one holding a list of input data, and the other holding a list of correct
        classifications in one-hot encoding form. Main then builds the ANN by calling
        the build function and then trains it, running back-propagation 2000 times on
        the training data while simultaneously using the validating dataset to validate
        it and calculate accuracy and loss values. Lastly, to simulate user testing,
        the testing dataset is passed in. Testing results from user input are then 
        printed out including the test data accuracy and all loss
        

    In terms of testing, we have segmented a portion of the data to simulate user
    input as asked for in the spec. However, if the user would like to test on
    additional custom input, they would be able to call the test function on a list
    containing lists of valid input and a list of the labels that correspond with
    them, receiving the accuracy and loss values for that custom testing set. However,
    our use of some of the data for testing allows for ease of testing with independent
    verifiable data without the need for user input of values and the corresponding
    correct classification