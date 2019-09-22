import numpy as np
import pandas
import math
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

def l_1_norm(x1, x2):
    '''
    compute the l-1 norm of a vector x2 with vector x1
    note: if the vectors are different length (which they shouldn't be),
        we take the smallest length of vector
    '''
    return np.linalg.norm([x1-x2], ord=1)

def l_2_norm(x1, x2):
    '''
    compute the l-2 norm of a vector x2 with vector x1
    note: if the vectors are different lenght (which they shouldn't be),
        we take the smallest length of vector
    '''
    return np.linalg.norm([x1-x2], ord=2)

def l_inf_norm(x1, x2):
    '''
    compute the l-infinity norm of a vector x1 with vector x2
    
    param x1, x2: vectors of ints
    '''
    return np.linalg.norm([x1-x2], ord=np.inf)

def rmse(y_estimates, y_valid):
    '''
    calculate the root mean squared error between the estimated y values and 
        the actual y values

    param y_estimates: list of ints, estimated y values by k-NN algorithm
    param y_valid: list of ints, actual y values
    return: float, the root mean squared error of the k-NN prediction 
    '''
    return np.sqrt(np.average(np.abs(y_estimates-y_valid)**2))

def five_fold_regression(x_train, x_valid, y_train, y_valid, distance_functions, k_list=None):
    '''
    take the data we know as test and valid, select a new section to be the 
        validation set, and then return the newly split data

    param distance_functions: list of function names to be iterated through
    param k: list of ints, k values to be used, if left as None, will iterate from
        1 -> 20
    return: list of errors by k and distance function: (k, dist_function, RMSE error)
    '''
    x_train_valid = np.vstack([x_train, x_valid])
    y_train_valid = np.vstack([y_train, y_valid])

    assert(len(x_train_valid) == len(y_train_valid))
    np.random.seed(5)
    np.random.shuffle(x_train_valid)
    np.random.seed(5)
    np.random.shuffle(y_train_valid)

    if not k_list:
        k_list = list(range(0,20))

    final_errors = []
    rmse_errors = {}                # contains RMSE errors sorted by k & distance function for each fold

    len_valid = len(x_train_valid)//5

    for i in range(5):
        x_valid = x_train_valid[i*len_valid:(i+1)*len_valid]
        x_train = np.vstack([x_train_valid[:i*len_valid], x_train_valid[(i+1)*len_valid:]])
        y_valid = y_train_valid[i*len_valid:(i+1)*len_valid]
        y_train = np.vstack([y_train_valid[:i*len_valid], y_train_valid[(i+1)*len_valid:]])

        for f in distance_functions:
            y_estimates = {}
            # assert(len(x_train) == len(y_train))

            for j in range(len(x_valid)):
                distances = []
                for p in range(len(x_train)):
                    dist = f(x_train[p], x_valid[j])
                    distances.append((dist, y_train[p]))

                distances.sort(key = lambda x: x[0])

                for k in k_list:
                    y = []
                    # append all the y values to a vector called y (stored in a dict)
                    for pair in distances[:k+1]:
                        y.append(pair[1])
                    avg_y = sum(y)/len(y)
                    
                    if k+1 not in y_estimates:
                        y_estimates[k+1] = []
                    y_estimates[k+1].append(avg_y)

            for k in k_list:
                # average errors should be a dictionary with values of list length 5 (i.e. 
                # one for each fold)
                if (k+1, f) not in rmse_errors:
                    rmse_errors[(k+1, f)] = []

                avg_rm_err = rmse(y_estimates[k+1], y_valid)
                rmse_errors[(k+1, f)].append(avg_rm_err)

    for k, func in rmse_errors:
        # take average RMSE across 5 folds
        final_error = sum(rmse_errors[(k, func)])/len(rmse_errors[(k, func)])   # length should be 5, for 5 folds
        # print('k: ' + str(k) + ' distance func: ' + str(func))
        # print('error: ' + str(final_error))
        final_errors.append((k, func, final_error))

    # average error
    return final_errors

def test_regression(x_train, x_valid, x_test, y_train, y_valid, y_test, k, func, plot=False):
    '''
    function only used for mauna loa set, to find the prediction on the test
        data and plot it, then find the cross validation loss.

    params data: all raw data as loaded from the data_utils file
    param k: optimal k value calculated for main
    param func: optimal distance function from main (should always be L2 for Mauna Loa)
    param plot: bool, only True of mauna loa - to create the plot
    '''
    x_train_valid = np.vstack([x_train, x_valid])
    y_train_valid = np.vstack([y_train, y_valid])

    y_pred = []

    for elem in x_test:
        d = []
        for i in range(len(x_train_valid)):
            # compute all point distances
            d.append((func(elem, x_train_valid[i]), y_train_valid[i]))

        d.sort(key=lambda x: x[0])

        y_est = 0
        for item in d[:k]:
            y_est += item[1]
        avg = y_est/k

        # save all y predictions to compute the root mean squared error after
        y_pred.append(avg)

    test_error = rmse(y_test, y_pred)

    if plot:
        # only enters this section for mauna loa dataset
        plt.figure(2)
        plt.plot(x_test, y_test, '-b', label='Actual')
        plt.plot(x_test, y_pred, '-r', label='Prediction')
        plt.title('Test predictions for Mauna Loa dataset')
        plt.xlabel('x test')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.savefig('mauna_loa_prediction.png')

    return test_error

def one_fold_classification(x_train, y_train, x_valid, y_valid, distance_functions, k_list=None):
    '''
    run the five fold training on all the classification datasets

    param x_train, y_train, x_valid, y_valid: data imported from data_utils
    param distance_functions: list of function names 
    param k_list: list of ints, k values to attempt, if left as default, will be filled with 1->20
    '''
    if not k_list:
        k_list = list(range(0,20))      # actual k values are k_list + 1

    correct = {}            # dictionary with k as keys, and with 

    for f in distance_functions:

        for s in range(len(x_valid)):
            distances = []
            # compute all distances for each point in this fold once
            for j in range(len(x_train)):
                dist = f(x_train[j], x_valid[s])
                distances.append((dist, y_train[j]))

            distances.sort(key = lambda x: x[0])

            # extract the k closest y points from the distances list so we can compute
            # the most commonly occuring one
            for k in k_list:
                y = []
                # append all the y values to a vector called y (stored in a dict)
                for d in distances[:k+1]:
                    y.append(d[1])
                
                # find the most commonly occuring class
                # length of y is at most 20
                oc_dct = {}
                assert(len(y) == k+1)
                
                # need to count the y points
                for estimate_pt in y:
                    if str(estimate_pt) not in oc_dct:
                        oc_dct[str(estimate_pt)] = (estimate_pt, 0)
                    oc_dct[str(estimate_pt)] = (estimate_pt, oc_dct[str(estimate_pt)][1] + 1)
                
                # sort the occurences to determine most common
                occurences = list(oc_dct.values())
                occurences.sort(key=lambda x: x[1], reverse=True)

                y_estimate = occurences[0][0]

                if np.all(y_estimate == y_valid[s]):
                    if (k+1, f) not in correct:
                        correct[(k+1, f)] = 0
                    correct[(k+1, f)] += 1

    # for each k value after the 5 folds, calculate the ratio of correctness to total points
    final_ratios = []
    for k, func in correct:
        ratio = correct[(k, func)]/len(x_valid)         # computes the % accuracy 
        # print('k: ' + str(k) + ' distance func: ' + str(func))
        # print('accuracy: ' + str(ratio))
        final_ratios.append((k, func, ratio))

    return final_ratios

def test_classification(x_train, x_valid, x_test, y_train, y_valid, y_test, k, distance_func):
    '''
    runs the test set using the x_train + x_valid to calculate the estimates

    param data: from the .npz data files
    param k: optimal k value calculated for main
    param distance_func: optimal distance function from main
    '''
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    y_pred = []
    correct = 0

    for s in range(len(x_test)):
        distances = []
        oc_dct = {}
        for i in range(len(x_train)):
            dist = distance_func(x_train[i], x_test[s])
            distances.append((dist, y_train[i]))
        distances.sort(key = lambda x: x[0])

        k_dist = distances[:k]

        assert(len(k_dist) == k)

        # need to count the y points
        for dist, estimate_pt in k_dist:
            if str(estimate_pt) not in oc_dct:
                oc_dct[str(estimate_pt)] = (estimate_pt, 0)
            oc_dct[str(estimate_pt)] = (estimate_pt, oc_dct[str(estimate_pt)][1] + 1)
        
        # sort the occurences to determine most common
        occurences = list(oc_dct.values())
        occurences.sort(key=lambda x: x[1], reverse=True)

        y_estimate = occurences[0][0]

        if np.all(y_estimate == y_test[s]):
            correct += 1

    # return accuracy (%)
    return correct/len(x_test)

def main(dataset, regression=True):
    '''
    runs one instance of kNN for 1 dataset at a time

    param dataset: str, name of dataset to be loaded, must be part of the 
        possible_datasets list
    param distance_func: function name, the distance function to be used in 
        neighbour calculation
    param regression: bool, true if regression, false if classification
    return: lowest k value and it's associated RMSE error
    '''
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(dataset)
    
    distance_functions = [l_1_norm, l_2_norm, l_inf_norm]
    if regression:
        results = []
        print('#######################')
        if dataset == 'mauna_loa':
            distance_functions = [l_2_norm]
        results = five_fold_regression(x_train, x_valid, y_train, y_valid, distance_functions)
        results.sort(key = lambda x: x[2])
        # results is list: (k, dist function, avg error over 5 folds)
        print('Dataset: ' + str(dataset))
        print('Best k:' + str(results[0][0]) + ' dist function: ' + str(results[0][1]))
        print('RMSE error: ' + str(results[0][2]))

        if dataset == 'mauna_loa':
            test_error = test_regression(x_train, x_valid, x_test, y_train, y_valid, y_test, results[0][0], results[0][1], True)

            # make required plots
            k_range = []
            plot_data = []
            results.sort(key = lambda x: x[0])      # sort by k for plotting
            for k, f, err in results:
                k_range.append(k)
                plot_data.append(err)

            plt.figure(1)
            plt.plot(k_range, plot_data)
            plt.title('Cross-validation error across k for Mauna Loa')
            plt.xlabel('k')
            plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
            plt.ylabel('RMSE Error')
            plt.savefig('mauna_loa_k.png')
        
        else:
            test_error = test_regression(x_train, x_valid, x_test, y_train, y_valid, y_test, results[0][0], results[0][1], False)
        print('Test error: ' + str(test_error))

    else:
        print('#######################')
        results = one_fold_classification(x_train, y_train, x_valid, y_valid, distance_functions)
        results.sort(key = lambda x: x[2], reverse=True)        # higher accuracy here means better
        print('Dataset: ' + str(dataset))
        print('Best k: ' + str(results[0][0]) + ' distance_function: ' + str(results[0][1]))
        print('Accuracy: ' + str(results[0][2]))
        # still need to implement the test portion of the algorithm 
        test_acc = test_classification(x_train, x_valid, x_test, y_train, y_valid, y_test, results[0][0], results[0][1])
        print('Test accuracy: ' + str(test_acc))

    # results are already sorted
    return results[0][0], results[0][2]

def test():
    '''
    function that can be called to run "unit tests" on all subfunctions 
        of the program
    '''
    # tests for finding the k_nearest neighbours
    test = [2,5,6]
    train = [[3,4,8], [10,9,6], [17,1,3], [1,2,3], [8,4,2], [5,11,9]]
    n = k_neighbours(l_2_norm, train, test, 2)
    print(n)
    assert(n == [[3,4,8], [1,2,3]])

    y_estimates = [1.309, -1.56, 1.45, 2.67, 3]
    y_valid = [4, -2, 6, 2, 2]
    error = rmse(y_estimates, y_valid)
    print(error)            # 2.4325
    error2 = rmse([], [])
    assert(error2 == 0)

if __name__ == "__main__":
    # should only run main or test, depending on the purpose of the run
    
    # 1. regression
    for set_i in possible_datasets[0:1]:
        k, distance_func = main(set_i, True)

    # 2. classification
    for set_i in possible_datasets[3:3]:
        k, distance_func = main(set_i, False)
    
    # test()