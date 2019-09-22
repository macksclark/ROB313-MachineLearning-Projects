import time
import numpy as np
import heapq
import math
from sklearn import neighbors
from matplotlib import pyplot as plt
from data.data_utils import load_dataset

__author__ = "Mackenzie Clark"
__date__ = "Feb. 12th 2019"

possible_datasets = ['rosenbrock']

def load_data(dataset, d):
    '''
    load the rosenbrock data to test the vectorization
    '''
    if dataset in possible_datasets:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset, n_train=5000, d=d)
        return x_train, x_valid, x_test, y_train, y_valid, y_test
    else:
        return 0,0,0,0,0,0 

def l_2_norm(x1, x2):
    '''
    compute the l-2 norm of a vector x2 with vector x1
    note: if the vectors are different lenght (which they shouldn't be),
        we take the smallest length of vector
    '''
    return np.linalg.norm([x1-x2], ord=2)

def rmse(y_estimates, y_valid):
    '''
    calculate the root mean squared error between the estimated y values and 
        the actual y values

    param y_estimates: list of lists, estimated y values by k-NN algorithm
    param y_valid: list of lists, actual y values
    return: float, the root mean squared error of the k-NN prediction 
    '''
    return np.sqrt(np.average(np.abs(y_estimates-y_valid)**2))

def part_a(x_train_valid, y_train_valid, x_test, y_test, distance_function, k):
    '''
    brute force algorithm for computing the y estimations on the test data for 
        rosenbrock dataset

    param train_valid x/y: merge between the validation and training data
    param test x/y: the test data to be used
    param distance_function: function name, to be used to calculate distance between 2 points
    param k: given k value to run k-NN with
    return: tuple, RMSE of regression results, and time taken to evaluate
    '''
    start = time.time()

    y_pred = []
    for pt in x_test:
        distances = []
        for i in range(len(x_train_valid)):
            dist = distance_function(x_train_valid[i], pt)
            distances.append((dist, y_train_valid[i]))
        distances.sort(key = lambda x: x[0])

        k_dist = []
        for el in distances[:k]:
            k_dist.append(el[1])

        avg = sum(k_dist)/len(k_dist)             # len(k_dist) will be k
        y_pred.append(avg)

    # compute RMSE between prediction and actual y point
    final_error = rmse(y_test, y_pred)

    end = time.time()
    print('Total time for part a: ' + str(end-start))
    print('RMSE error: ' + str(final_error))
    return (end-start, final_error)

def part_b(x_train_valid, y_train_valid, x_test, y_test, k):
    '''
    partially vectorized code for the rosenbrock dataset test data

    param train_valid x/y: merge between the validation and training data
    param test x/y: the test data to be used
    param distance_function: function name, to be used to calculate distance between 2 points
    param k: given k value to run k-NN with
    return: tuple, RMSE of regression results, and time taken to evaluate
    '''
    start = time.time()
    avg_rmse = np.zeros(len(x_test))

    pred = []
    for i in range(len(x_test)):
        res = np.sqrt(np.sum(np.square(x_train_valid-x_test[i]), axis=1))
        k_smallest = heapq.nsmallest(k, range(len(res)), res.take)
        pred.append(np.array([np.average(np.take(y_train_valid, k_smallest))]))

    # calculate RMSE error between avg point and actual y_test
    final_error = rmse(y_test, np.array(pred))

    end = time.time()
    print('Total time for part b: ' + str(end-start))
    print('RMSE error: ' + str(final_error))
    return (end-start, final_error)

def part_c(x_train_valid, y_train_valid, x_test, y_test, k):
    '''
    fully vectorized code, the function will expand the dimensions of the x_train_valid
        and then store the expanded

    param train_valid x/y: merge between the validation and training data
    param test x/y: the test data to be used
    param distance_function: function name, to be used to calculate distance between 2 points
    param k: given k value to run k-NN with
    return: tuple, RMSE of regression results, and time taken to evaluate
    '''
    start = time.time()
    # create the distance matrix d
    d = np.sqrt(-2 * np.dot(x_test, x_train_valid.T) + np.sum(x_train_valid ** 2, axis=1) + np.sum(x_test ** 2, axis=1)[:, np.newaxis])
    k_nb = np.argpartition(d, kth=k, axis=1)[:, :k]
    y_pred = np.sum(y_train_valid[k_nb],axis=1)/k

    final_error = rmse(y_test, y_pred)
    end = time.time()
    print('Total time for part c: ' + str(end-start))
    print('RMSE error: ' + str(final_error))
    return (end-start, final_error)

def part_d(x_train_valid, y_train_valid, x_test, y_test, k):
    '''
    k-d tree implementation of calculating the k-NN algorithm

    param train_valid x/y: merge between the validation and training data
    param test x/y: the test data to be used
    param distance_function: function name, to be used to calculate distance between 2 points
    param k: given k value to run k-NN with
    return: tuple, RMSE of regression results, and time taken to evaluate
    '''
    start = time.time()
    kdt = neighbors.KDTree(x_train_valid)
    d, k_nb = kdt.query(x_test, k=k)
    y_predictions = np.sum(y_train_valid[k_nb],axis=1)/k

    final_error = rmse(y_test, y_predictions)
    end = time.time()
    print('Total time for part d: ' + str(end-start))
    print('RMSE error: ' + str(final_error))
    return (end-start, final_error)

if __name__ == "__main__":
    # main block to run the vectorization code
    k = 5
    distance_function = l_2_norm

    d_range = list(range(2,10))
    times = {'a': [], 'b': [], 'c': [], 'd': []}
    rmse_t = {'a': [], 'b': [], 'c': [], 'd': []}
    for d in range(2,10):
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data('rosenbrock', d)
    
        # initializing variables 
        x_train_valid = np.vstack([x_train, x_valid])
        y_train_valid = np.vstack([y_train, y_valid])

        # part a - brute force
        timer, error = part_a(x_train_valid, y_train_valid, x_test, y_test, distance_function, k)
        times['a'].append(timer)
        rmse_t['a'].append(error)

        # run part b - partial vectorization
        timer, error = part_b(x_train_valid, y_train_valid, x_test, y_test, k)
        times['b'].append(timer)
        rmse_t['b'].append(error)

        # run part c - full vectorized code (no loops)
        timer, error = part_c(x_train_valid, y_train_valid, x_test, y_test, k)
        times['c'].append(timer)
        rmse_t['c'].append(error)

        # run part d - k-d data structure for k-NN
        timer, error = part_d(x_train_valid, y_train_valid, x_test, y_test, k)
        times['d'].append(timer)
        rmse_t['d'].append(error)

    plt.figure(1)
    plt.title('Rosenbrock n=5000 runtimes for varying d')
    plt.xlabel('d')
    plt.ylabel('Runtime [s]')
    plt.plot(d_range, times['a'], '-m', label='part a')
    plt.plot(d_range, times['b'], '-g', label='part b')
    plt.plot(d_range, times['c'], '-y', label='part c')
    plt.plot(d_range, times['d'], '-r', label='part d')
    plt.legend()
    plt.savefig('part3_runtime.png')

    plt.figure(2)
    plt.title('Rosenbrock n=5000 RMSE for varying d')
    plt.xlabel('d')
    plt.ylabel('RMSE')
    plt.plot(d_range, rmse_t['a'], '-m', label='part a')
    plt.plot(d_range, rmse_t['b'], '-g', label='part b')
    plt.plot(d_range, rmse_t['c'], '-y', label='part c')
    plt.plot(d_range, rmse_t['d'], '-r', label='part d')
    plt.legend()
    plt.savefig('part3_rmse.png')