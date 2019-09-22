__author__ = "Mackenzie Clark"
__date__ = "Mar 28th 2019"

import autograd.numpy as np
from autograd import value_and_grad
from data.data_utils import load_dataset
from matplotlib import pyplot as plt
plt.switch_backend('agg')

possible_datasets = ["mnist_small"]

def forward_pass(W1, W2, W3, b1, b2, b3, x):
    """
    forward-pass for an fully connected neural network with 2 hidden layers of M neurons
    Inputs:
        W1 : (M, 784) weights of first (hidden) layer
        W2 : (M, M) weights of second (hidden) layer
        W3 : (10, M) weights of third (output) layer
        b1 : (M, 1) biases of first (hidden) layer
        b2 : (M, 1) biases of second (hidden) layer
        b3 : (10, 1) biases of third (output) layer
        x : (N, 784) training inputs
    Outputs:
        Fhat : (N, 10) output of the neural network at training inputs
    """
    H1 = np.maximum(0, np.dot(x, W1.T) + b1.T) # layer 1 neurons with ReLU activation, shape (N, M)
    H2 = np.maximum(0, np.dot(H1, W2.T) + b2.T) # layer 2 neurons with ReLU activation, shape (N, M)
    Fhat = np.dot(H2, W3.T) + b3.T # layer 3 (output) neurons with linear activation, shape (N, 10)
    
    a = np.max(Fhat, axis=1)
    # expand the value of a into a matrix to compute log-exp trick easy
    A = -1*np.ones(np.shape(Fhat))*a[:, np.newaxis]

    log_sum = np.log(np.sum(np.exp(np.add(Fhat, A)), axis=1))
    # turn sums into a matrix as well
    Log_Sum = -1*np.ones(np.shape(Fhat))*log_sum[:, np.newaxis]

    # log-exp trick on each element of Fhat
    Fhat = np.add(np.add(Fhat, A), Log_Sum)
    return Fhat

def negative_log_likelihood(W1, W2, W3, b1, b2, b3, x, y):
    """
    computes the negative log likelihood of the model `forward_pass`
    Inputs:
        W1, W2, W3, b1, b2, b3, x : same as `forward_pass`
        y : (N, 10) training responses
    Outputs:
        nll : negative log likelihood
    """
    Fhat = forward_pass(W1, W2, W3, b1, b2, b3, x)

    nll = np.einsum('ij, ij->i', Fhat, y)
    return -1*np.sum(nll)

nll_gradients = value_and_grad(negative_log_likelihood, argnum=[0,1,2,3,4,5])
"""
    returns the output of `negative_log_likelihood` as well as the gradient of the 
    output with respect to all weights and biases
    Inputs:
        same as negative_log_likelihood (W1, W2, W3, b1, b2, b3, x, y)
    Outputs: (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad))
        nll : output of `negative_log_likelihood`
        W1_grad : (M, 784) gradient of the nll with respect to the weights of first (hidden) layer
        W2_grad : (M, M) gradient of the nll with respect to the weights of second (hidden) layer
        W3_grad : (10, M) gradient of the nll with respect to the weights of third (output) layer
        b1_grad : (M, 1) gradient of the nll with respect to the biases of first (hidden) layer
        b2_grad : (M, 1) gradient of the nll with respect to the biases of second (hidden) layer
        b3_grad : (10, 1) gradient of the nll with respect to the biases of third (output) layer
     """
    
def run_example():
    """
    This example demonstrates computation of the negative log likelihood (nll) as
    well as the gradient of the nll with respect to all weights and biases of the
    neural network. We will use 50 neurons per hidden layer and will initialize all 
    weights and biases to zero.
    """
    # load the MNIST_small dataset
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    
    # initialize the weights and biases of the network
    M = 50 # 50 neurons per hidden layer
    W1 = np.zeros((M, 784)) # weights of first (hidden) layer
    W2 = np.zeros((M, M)) # weights of second (hidden) layer
    W3 = np.zeros((10, M)) # weights of third (output) layer
    b1 = np.zeros((M, 1)) # biases of first (hidden) layer
    b2 = np.zeros((M, 1)) # biases of second (hidden) layer
    b3 = np.zeros((10, 1)) # biases of third (output) layer
    
    # considering the first 250 points in the training set, 
    # compute the negative log likelihood and its gradients
    (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
        nll_gradients(W1, W2, W3, b1, b2, b3, x_train[:250], y_train[:250])
    print("negative log likelihood: %.5f" % nll)

def load_data(dataset):
    '''
    load the data using the function from data_utils.py
    '''
    if dataset not in possible_datasets:
        return 0,0,0,0,0,0
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def Xavier_init(hidden_neurons, output_neurons, n_inputs):
    '''
    calculate the weights for initialization using the
    Xavier initialization method

    param hidden_neurons: int, number of neurons for all hidden layers
    param output_neurons: int, number of neurons in the output layer
    param n_inputs: number of features in the input to the first layer
    '''
    W1 = np.random.randn(hidden_neurons, n_inputs)/np.sqrt(n_inputs)
    W2 = np.random.randn(hidden_neurons, hidden_neurons)/np.sqrt(hidden_neurons)
    W3 = np.random.randn(output_neurons, hidden_neurons)/np.sqrt(hidden_neurons)
    b1 = np.zeros((hidden_neurons, 1))
    b2 = np.zeros((hidden_neurons, 1))
    b3 = np.zeros((output_neurons, 1))
    return W1, W2, W3, b1, b2, b3

def update_weights(weights, grad_weights, rate):
    '''
    given the weight gradients from nll_gradients(), update the weights
    by the amount rate

    param weights: list, old weight values
    param grad_weights: list, weight gradients
    param rate: float, learning rate
    '''
    output = []
    for i in range(len(weights)):
        output.append(np.subtract(weights[i], rate*grad_weights[i]))
    return output

def update_bias(bias, grad_bias, rate):
    '''
    given the bias gradients from nll_gradients(), update the bias values
    by the amount rate

    param bias: list, old bias values
    param grad_bias: list, bias gradients
    param rate: float, learning rate
    '''
    output = []
    for i in range(len(bias)):
        output.append(np.subtract(bias[i], rate*grad_bias[i]))
    return output

def calculate_accuracy(weights, bias, x, y):
    '''
    given the minimizing weights and bias, compute the accuracy for x_valid and y_valid
    '''
    Fhat = np.exp(forward_pass(weights[0], weights[1], weights[2], bias[0], bias[1], bias[2], x))
    y_pred = np.argmax(Fhat, axis=1)
    y = np.argmax(y, axis=1)
    return (y_pred == y).sum() / len(y)

def neural_network(x_train, x_valid, x_test, y_train, y_valid, y_test, weights, bias, batch_size, test=False, acc=False, \
    visual=False, digit=False, iterations=1000, rates=[0.0001, 0.001]):
    '''
    trains a neural network for the data in x_train, and computes the 
    training and validation error per iteration

    param data: from the data_utils.py script
    param weights: list of W, initial weight vector guesses
    param bias: list of b, initial bias guesses
    param batch_size: int, number of random elements to include in the mini-batch
    param test: bool, True if we should calculate the test error for the set
    param acc: bool, True if we want to calculate the accuracy of the model
    param visual: bool, True if we want to visualize 16 of the weights
    param iterations: int, number of iterations of GD to execute
    param rates: list of floats, learning rates to train the model with GD
    '''
    W1 = weights[0]
    W2 = weights[1]
    W3 = weights[2]
    b1 = bias[0]
    b2 = bias[1]
    b3 = bias[2]

    train_nll = {}
    valid_nll = {}
    valid_acc = {}
    test_nll = {}
    test_acc = {}
    min_valid_nll = {}
    min_nll_it = {}

    for rate in rates:
        train_nll[rate] = []
        valid_nll[rate] = []

        min_nll = np.inf
        min_nll_weights = None
        min_nll_bias = None

        for i in range(iterations):
            # compute mini-batch gradient descent to estimate the new weights
            batch_indices = np.random.choice(np.shape(x_train)[0], size=batch_size, replace=False)
            x_mini_batch = x_train[batch_indices, :]
            y_mini_batch = y_train[batch_indices, :]

            # use the autograd nll function
            (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
                nll_gradients(W1, W2, W3, b1, b2, b3, x_mini_batch, y_mini_batch)
            
            # calculate the full-batch validation neg. ll at every iteration
            cur_valid_nll = negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
            valid_nll[rate].append(cur_valid_nll)

            # use the mini-batch neg. log likelihood but scaled to the size of x_train (NORMALIZED)
            train_nll[rate].append(nll*len(x_train)/batch_size)
            # compute the minimum neg. ll to use this number of iterations on the test set
            if cur_valid_nll < min_nll:
                min_nll = cur_valid_nll
                min_nll_weights = [W1, W2, W3]
                min_nll_bias = [b1, b2, b3]
                # dictionaries we need to return
                min_valid_nll[rate] = cur_valid_nll
                min_nll_it[rate] = i+1

            # calculate new weights
            [W1, W2, W3] = update_weights([W1, W2, W3], [W1_grad, W2_grad, W3_grad], rate)
            [b1, b2, b3] = update_bias([b1, b2, b3], [b1_grad, b2_grad, b3_grad], rate)

        if test:
            # use early stopping for the test set based on the minimum validation error
            test_nll[rate] = negative_log_likelihood(min_nll_weights[0], min_nll_weights[1], \
                min_nll_weights[2], min_nll_bias[0], min_nll_bias[1], min_nll_bias[2], x_test, y_test)
            # use the min validation error iteration to compute accuracy (test and validation)
            valid_acc[rate] = calculate_accuracy(min_nll_weights, min_nll_bias, x_valid, y_valid)
            test_acc[rate] = calculate_accuracy(min_nll_weights, min_nll_bias, x_test, y_test)

        if digit:
            Fhat = np.max(np.exp(forward_pass(min_nll_weights[0], min_nll_weights[1], \
                min_nll_weights[2], min_nll_bias[0], min_nll_bias[1], min_nll_bias[2], x_test)), axis=1)
            sorted_ind = np.argsort(Fhat)
            sorted_test_set = x_test[sorted_ind]

    if visual:
        # visualize 16 random weights for the first layer of the network 
        M = len(W1)
        for i in range(17):
            j = np.random.randint(M)
            plot_digit_mod(W1[j], i + 10, "weight_vis_" + str(j))
    elif digit:
        # sorted ind and sorted_test_set exist, so we will plot the figures
        for i in range(10):
            plot_digit_mod(sorted_test_set[i], i + 26, "test_" + str(sorted_ind[i]) + "rank_" + str(i))

    return train_nll, valid_nll, valid_acc, test_nll, test_acc, min_valid_nll, min_nll_it

def plot_nll(training_data, valid_data, rate, neurons, figure_num):
    '''
    takes the data of neg. log-likelihood per iteration and the rate at
    which the algorithm was executed, then plots the nll vs. iteration
    
    param training_nll: list, neg. ll values from training
    param valid_nll: list, neg. ll values from validation set testing
    param mb_size: int, minibatch size
    param figure_num: int, unique number to define each figure
    '''
    x = range(1, len(training_data)+1)
    # normalize the values
    training_data = np.divide(training_data, 10000)
    valid_data = np.divide(valid_data, 1000)

    plt.figure(figure_num)
    plt.title("SGD (neurons=" + str(neurons)+ ") neg. log-likelihood, learning rate = " + str(rate))
    plt.xlabel("Iteration")
    plt.ylabel("Neg. log-likelihood")
    plt.plot(x, training_data, '-b', label="Training nll")
    plt.plot(x, valid_data, '-r', label="Validation nll")
    plt.legend(loc='upper right')
    plt.savefig("nll_size_" + str(neurons) + "_rate_" + str(rate) + ".png")

def plot_digit_mod(x, figure_num, title):
    '''
    plots a provided mnist digit given by x
    '''
    assert np.size(x) == 784
    x.reshape(28, 28)
    plt.figure(figure_num)
    plt.imshow(x.reshape((28, 28)), interpolation='none', aspect='equal', cmap='gray')
    plt.savefig(title + ".png")

if __name__ == "__main__":
    # set these params to true if you want to run their respective questions 
    Q3 = False
    Q4 = False
    Q5 = False
    Q6 = True 

    i = 1
    if Q3:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data('mnist_small')
        W1, W2, W3, b1, b2, b3 = Xavier_init(100, 10, len(x_train[0]))
        weights_init = [W1, W2, W3]
        bias_init = [b1, b2, b3]
        training_nll, valid_nll, valid_acc, junk, junk2, min_valid, min_it = neural_network(x_train, x_valid, x_test, y_train, y_valid, y_test, \
            weights_init, bias_init, 250, iterations=4000, rates=[0.0001, 0.0005])
        
        # plot all the figures
        for r in training_nll:
            print("----------------- neurons: 100 ---------------")
            print("rate: " + str(r))
            print("best validation neg. LL: " + str(min_valid[r]) + ", at iteration: " + str(min_it[r]))
            plot_nll(training_nll[r], valid_nll[r], r, 100, i)
            i += 1

    if Q4:
        # create the neural network with multiple different hidden level num. nodes, keeping the batch size still 250
        nn_sizes = [10, 120, 200]
        it = {10: 4000, 120: 3000, 200: 3000}
        rat = {10: [0.0001, 0.0005, 0.001], 120: [0.0001, 0.0005, 0.001], 200: [0.0001, 0.0005, 0.001]}
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data('mnist_small')

        for s in nn_sizes:
            W1, W2, W3, b1, b2, b3 = Xavier_init(s, 10, len(x_train[0]))
            weights_init = [W1, W2, W3]
            bias_init = [b1, b2, b3]
            training_nll, valid_nll, valid_acc, test_nll, test_acc, min_valid, min_it = neural_network(x_train, x_valid, x_test, y_train, y_valid, y_test,\
                weights_init, bias_init, 250, True, True, iterations=it[s], rates=rat[s])

            # print the best results and plot the figures for visualization (not necessary for assignment)
            for r in training_nll:
                print("----------------- neurons: " + str(s) + " ---------------")
                print("rate: " + str(r))
                print("best validation neg. LL: " + str(min_valid[r]) + ", at iteration: " + str(min_it[r]))
                print("best validation accuracy: " +str(valid_acc[r]))
                print("test neg. LL: " + str(test_nll[r]))
                print("test accuracy: " + str(test_acc[r]))
                # plot_nll(training_nll[r], valid_nll[r], r, s, i)
                i += 1

    if Q5:
        # create the neural network and create figures for 16 of the weights
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data('mnist_small')
        W1, W2, W3, b1, b2, b3 = Xavier_init(100, 10, len(x_train[0]))
        weights_init = [W1, W2, W3]
        bias_init = [b1, b2, b3]
        junk = neural_network(x_train, x_valid, x_test, y_train, y_valid, y_test,\
            weights_init, bias_init, 250, visual=True, iterations=1000, rates=[0.0001])

    if Q6:
        # create the graphs for 
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data('mnist_small')
        W1, W2, W3, b1, b2, b3 = Xavier_init(100, 10, len(x_train[0]))
        weights_init = [W1, W2, W3]
        bias_init = [b1, b2, b3]
        junk = neural_network(x_train, x_valid, x_test, y_train, y_valid, y_test,\
            weights_init, bias_init, 250, digit=True, iterations=1000, rates=[0.0001])