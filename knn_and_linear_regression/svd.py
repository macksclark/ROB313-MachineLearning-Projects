import numpy as np
import time
from matplotlib import pyplot as plt
from data.data_utils import load_dataset

__author__ = "Mackenzie Clark"
__date__ = "Feb. 12th 2019"

possible_datasets = ['mauna_loa', 'rosenbrock', 'pumadyn32nm', 'iris', 'mnist_small']

def load_data(dataset):
    '''
    param dataset: str, the name of the dataset to be loaded for this iteration of the model
    '''
    if dataset not in possible_datasets:
        return 0,0,0,0,0,0
    elif dataset == 'rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=1000, d=2)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(str(dataset))
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def rmse(y_estimates, y_valid):
    '''
    calculate the root mean squared error between the estimated y values and 
        the actual y values

    param y_estimates: list of lists, estimated y values by k-NN algorithm
    param y_valid: list of lists, actual y values
    return: float, the root mean squared error of the k-NN prediction 
    '''
    return np.sqrt(np.average(np.abs(y_estimates-y_valid)**2))

def svd_regression(dataset):
    '''
    compute the FULL singular value decomposition of the matrix 
        of x_train_valid values

    param dataset: str, dataset name, must be a part of possible_datasets
    '''
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(dataset)
    start = time.time()

    x_train_valid = np.vstack([x_train, x_valid])
    y_train_valid = np.vstack([y_train, y_valid])
    X = np.ones((len(x_train_valid), len(x_train_valid[0])+1))
    X[:, 1:] = x_train_valid

    # compute the matrices for SVD
    U, s, vh = np.linalg.svd(X, full_matrices=True)
    Sigma = np.diag(s)
    zero_m = np.zeros([len(x_train_valid)-len(Sigma), len(Sigma)])

    # concatenate the singular values to 0 matrix to be the same 
    # dimension as x_train_valid 
    S_full = np.vstack([Sigma, zero_m])
    # determining the weights 
    w = np.dot(vh.T, np.dot(np.linalg.pinv(S_full), np.dot(U.T, y_train_valid)))

    # copy over the values of x_test
    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test
    y_pred = np.dot(X_test, w)

    error = rmse(y_test, y_pred)
    end = time.time()

    if dataset == "mauna_loa":
        # only make plot for 1D mauna loa set as a test
        plt.figure(1)
        plt.plot(x_test, y_test, '-b', label='Actual')
        plt.plot(x_test, y_pred, '-r', label='Prediction')
        plt.title('SVD predictions for Mauna Loa dataset')
        plt.xlabel('x test')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.savefig('mauna_loa_svd.png')

    return (end-start, error)

def svd_classification(dataset):
    '''
    compute the one of k binary classification for the classification
        datasets using linear regression/SVD

    param dataset: str, dataset name, must be a part of possible_datasets
    '''
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(dataset)
    start = time.time()

    x_train_valid = np.vstack([x_train, x_valid])
    y_train_valid = np.vstack([y_train, y_valid])

    # add column of ones to X to account for w0
    X = np.ones([len(x_train_valid), len(x_train_valid[0]) + 1])
    X[:, 1:] = x_train_valid

    U, s, vh = np.linalg.svd(X)

    # calculate the sigma matrix
    Sigma = np.diag(s)
    zero_m = np.zeros([len(x_train_valid) - len(Sigma), len(Sigma)])
    S_full = np.vstack([Sigma, zero_m])

    # calculate weights for the SVD to get y predictions
    w = np.dot(vh.T, np.dot(np.linalg.pinv(S_full), np.dot(U.T, y_train_valid)))

    # copy over the values of x_test
    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test

    # find the maximum values of the predictions and the test,
    # to compare and calculate accuracy
    y_pred = np.argmax(np.dot(X_test, w), axis=1)
    y_test = np.argmax(1 * y_test, axis=1)

    # count the number of correct classifications
    result = (y_pred == y_test).sum() / len(y_test)

    end = time.time()
    return (end-start, result)  

if __name__ == "__main__":
    # 1. regression
    for set_i in possible_datasets[0:1]:
        timer, error = svd_regression(set_i)
        print('Dataset: ' + str(set_i) + ' ran in: ' + str(timer))
        print('RMSE: ' + str(error))

    # 2. classification
    for set_i in possible_datasets[3:3]:
        timer, error = svd_classification(set_i)
        print('Dataset: ' + str(set_i) + ' ran in: ' + str(timer))
        print('RMSE: ' + str(error))