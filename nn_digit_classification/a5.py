__author__ = "Mackenzie Clark"
__date__ = "Apr 13th 2018"

import numpy as np
from data.data_utils import load_dataset
from matplotlib import pyplot as plt
import scipy.stats
import math
plt.switch_backend('agg')

possible_datasets = ["iris"]

################ helper functions from previous assingments #################

def load_data():
    '''
    load the iris dataset using the specified method
    '''
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def make_x_matrix(x_data):
    '''
    takes a vetor of data (matrix) x_data and adds a column of 1's
    at the front
    '''
    X = np.ones((len(x_data), len(x_data[0]) + 1))
    X[:, 1:] = x_data
    return X

def sigmoid(z):
    '''
    calculate the sigmoid for a vector x
    '''
    sig = np.divide(1, np.add(1, np.exp(-1*z)))
    if str(type(sig)) != "<class 'numpy.float64'>":
        sig = np.reshape(sig, (np.shape(sig)[0], 1))
    return sig

##################### question 1 code ########################

def marginal_likelihoods(x_train, y_train, learning_rate=0.001, variances = [0.5, 1, 2]):
    '''
    compute the bayesian predictions

    param learning_rate: float, desired learning rate to compute MAP w with GD
    param variances: list of floats, variance of the gaussian prior dist.
    '''
    err = 10 ** (-6)
    iterations = {}
    marginal_likelihood = {}
    weights = {}
    x_train = make_x_matrix(x_train)

    for v in variances:
        # initialize the weights to 0
        w = np.zeros(np.shape(x_train[0]))
        iterations[v] = 0
        gradient = np.inf

        # compute the maximum a posteriori (MAP) estimate numerically
        while np.max(gradient) > err:

            train_pred = sigmoid(np.dot(x_train, w))
            gradient = np.add(likelihood_grad(train_pred, y_train, x_train), prior_grad(w, v)) 
            w = np.add(w, np.multiply(gradient, learning_rate))
            iterations[v] += 1

        # now with the MAP estimate, compute the hessian and do the laplace approximation
        weights[v] = w
        map_pred = sigmoid(np.dot(x_train, w))
        map_hessian = np.add(likelihood_hess(map_pred, x_train), prior_hess(w, v))

        # compute the laplace approximation at the MAP w value
        g_w =  len(w)/2*math.log(2*math.pi) - 1/2*math.log(np.linalg.det (-1 * map_hessian))
        marginal_likelihood[v] = -1* (log_likelihood(map_pred, y_train) + log_prior(w, v) + g_w)

    return marginal_likelihood, iterations, weights

def log_likelihood(f_hat, y):
    '''
    computes the log of the likelihood P(y|w,X)
    param f_hat: numpy vector, fhat (sigmoid of X (dot) w)
    param y: numpy vector, actual y
    '''
    return np.sum(np.add(np.multiply(y, np.log(f_hat)), np.multiply(np.subtract(1, y), np.log(np.subtract(1, f_hat)))))

def log_prior(w, sigma):
    '''
    computes the log of the prior P(w)
    param w: numpy vector, current weight estimates
    param sigma: float, variance of prior
    '''
    D = len(w)
    return - (D)/2 * (math.log(2*math.pi) + math.log(sigma)) - np.sum(np.divide(np.power(w, 2), 2*sigma))

def likelihood_grad(f_hat, y, x):
    '''
    computes the gradient of the likelihood P(y|w,X)
    '''
    gradient = np.zeros(np.shape(x[0]))
    v = np.subtract(y, f_hat)
    for i in range(len(x)):
        # compute the additive gradient
        gradient = np.add(gradient, v[i] * x[i])
    return gradient

def prior_grad(w, sigma):
    '''
    computes the gradient of the prior P(w)
    '''
    return np.divide(-1*w, sigma)

def likelihood_hess(f_hat, x):
    '''
    computes the hessian of the likelihood P(y|w,X)
    '''
    hessian = np.zeros((np.shape(x)[1], np.shape(x)[1]))
    v = np.multiply(f_hat, np.subtract(f_hat, 1))
    for i in range(len(x)):
        # compute xbar * xbar.T
        m = np.outer(x[i], x[i].T)
        hessian = np.add(hessian, v[i] * m)
    return hessian      # M x M matrix

def prior_hess(w, sigma):
    '''
    computes the hessian of the prior P(w)
    '''
    return np.eye(np.shape(w)[0]) * -1 / sigma

##################### question 2 code ########################

def importance_sampling(x_train, x_valid, x_test, y_train, y_valid, y_test, q_mean, S=[10, 100, 500]):
    '''
    using prior variance of 1, and importance sampling, to estimate the predictive posterior

    param q_mean: estimated mean of the Gaussian q(w) distribution, usually the Laplace Approx of the map
    param S: number of w values to sample from the proposal
    '''
    var = 1
    x_train = make_x_matrix(x_train)
    x_valid = make_x_matrix(x_valid)
    x_test = make_x_matrix(x_test)
    q_vars = [1, 2, 5]
    z = np.zeros(np.shape(q_mean))

    # perform cross validation for the sampling size S and the variance and mean of the q(w) function
    neg_lls = {}
    accuracies = {}

    for sample_size in S:
        for q_var in q_vars:

            predictions = np.ndarray(np.shape(y_valid))
            predictions_discrete = np.ndarray(np.shape(y_valid))

            # grab the weights and calculate the sum of r(w) values
            r_sum = 0           # store the sum of all the r(w) values
            weights = []
            priors = []
            lls = []
            qs = []
            for j in range(sample_size):
                cur_w = proposal(q_mean, q_var)
                weights.append(cur_w)
                pr = gaussian(cur_w, var, z)
                priors.append(pr)
                ll = likelihood(sigmoid(np.dot(x_train, cur_w)), y_train)
                lls.append(ll)

                q = gaussian(cur_w, q_var, q_mean)
                qs.append(q)
                r_sum += pr * ll / q

            for k in range(len(x_valid)):

                cur_pred = 0
                for i in range(sample_size):
                    # do the final sum
                    current_r = priors[i] * lls[i] / qs[i]
                    y_pred = sigmoid(np.dot(x_valid[k], weights[i])) 
                    # calculate the probability of prediction
                    cur_pred += y_pred * current_r / r_sum
                predictions[k] = cur_pred

                if cur_pred > 0.5:
                    predictions_discrete[k] = 1
                elif cur_pred < 0.5:
                    predictions_discrete[k] = 0
                else:
                    # if the P(y = 1) = P(y = 0) = 0.5
                    predictions_discrete[k] = -1

            # compute the negative log likelihood of the predictions and add it to the neg_lls dict
            # so that we can figure out which metrics are best
            neg_lls[(q_var, sample_size)] = -1 * log_likelihood(predictions, y_valid)
            accuracies[(q_var, sample_size)] = (predictions_discrete == y_valid).sum() / len(y_valid)

    print("----------- q2 -----------")
    print("Negative log-likelihoods: ")
    print(neg_lls)
    print("Accuracies: ")
    print(accuracies)
    # find the minimum of the negative log_likelihoods
    min_neg_ll = np.inf
    for v, s in neg_lls:
        if neg_lls[(v, s)] < min_neg_ll:
            min_neg_ll = neg_lls[(v, s)]
            min_v = v
            min_s = s

    print("best variance q(w): " + str(min_v))
    print("best s value: " + str(min_s))

    # merge the training and validation set so we can compute the 
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    test_pred = np.ndarray(np.shape(y_test))
    test_pred_discrete = np.ndarray(np.shape(y_test))

    # compute the predictions using min_s number of gaussian weight samples
    r_sum = 0           # store the sum of all the r(w) values
    weights = []
    priors = []
    lls = []
    qs = []
    # randomly grab the weights from the same distribution as the training ones
    for j in range(min_s):
        cur_w = proposal(q_mean, min_v)
        weights.append(cur_w)
        pr = gaussian(cur_w, var, z)
        priors.append(pr)
        ll = likelihood(sigmoid(np.dot(x_train, cur_w)), y_train)
        lls.append(ll)

        q = gaussian(cur_w, min_v, q_mean)
        qs.append(q)
        r_sum += pr * ll / q

    rs_for_plot = []
    ws_for_plot = []
    for k in range(len(x_test)):

        cur_pred = 0
        for i in range(min_s):
            # do the final sum
            current_r = priors[i] * lls[i] / qs[i]
            y_pred = sigmoid(np.dot(x_test[k], weights[i])) 
            # calculate the probability of prediction
            cur_pred += y_pred * current_r / r_sum
            rs_for_plot.append(current_r / r_sum)
            ws_for_plot.append(weights[i])

        test_pred[k] = cur_pred

        if cur_pred > 0.5:
            test_pred_discrete[k] = 1
        elif cur_pred < 0.5:
            test_pred_discrete[k] = 0
        else:
            # if the probability of being 1 is equal to -1
            test_pred_discrete[k] = -1

    # calculate the negative log likelihood and accuracy ratio for test set
    test_nll = -1 * log_likelihood(test_pred, y_test)
    test_acc = (test_pred_discrete == y_test).sum() / len(y_test)

    return test_nll, test_acc, min_v, rs_for_plot, ws_for_plot

def proposal(mean, var):
    '''
    compute the proposal distribution as a multivariate gaussian/normal 
    
    param mean: numpy array, the proposed mean of the gaussian
    param var: int, the proposed variance of the gaussian
    '''
    return np.random.multivariate_normal(mean=mean, cov=np.eye(np.shape(mean)[0])*var)

def gaussian(w, var, mean):
    '''
    compute the prior distribution of w

    param var: float, variance of the Gaussian prior distribution 
    param mean: numpy array, the mean of the gaussian used, for prior = 0 
    '''
    prior = 1
    for i in range(len(w)):
        prior = prior / math.sqrt(2 * math.pi * var) * math.exp(-1 * ((w[i] - mean[i])**2) / (2*var))
    return prior

def likelihood(y_pred, y):
    '''
    compute the likelihood function of the entire dataset predictions

    param y_pred: numpy array, SIGMOID prediction values
    param y: numpy array, actual y values
    '''
    ll = 1
    for i in range(len(y)):
        ll = ll * (y_pred[i] ** y[i]) * ((1 - y_pred[i]) ** (1 - y[i]))
    return ll

def visualization(mean, var, posterior, w):
    '''
    makes a graph using matplotlib with w vs. posterior (p(w|X,y))

    param mean: numpy array, the mean of q(w) (MAP)
    param var: int, the variance of q(w) (selected in cross-validation)
    param posterior: list, the posterior distribution calculated when sampling 
    param w: list, the weights sampled used when calculating the posterior distribution 
    '''
    for i in range(5):
        weights = []
        for j in range(len(w)):
            weights.append(w[j][i])

        # calculate the trendline
        z = np.polyfit(weights, posterior, 1)
        z = np.squeeze(z)
        p = np.poly1d(z)

        w_all = np.arange(min(weights), max(weights), 0.001)
        q_w = scipy.stats.norm.pdf(w_all, mean[i], var)
        plt.figure(i)
        plt.title("Posterior visualization: q(w) mean=" + str(round(mean[i], 2)) + " var=" + str(var))
        plt.xlabel("w[" + str(i) + "]")
        plt.ylabel("Probability")
        plt.plot(w_all, q_w, '-b', label="Proposal q(w)")
        plt.plot(weights, posterior, 'or', label="Posterior P(w|X,y)")
        plt.plot(weights, p(weights),"r--")
        plt.legend(loc='upper right')
        plt.savefig("weight_vis_" + str(i) + ".png")

##################### question 3 code ########################

def metropolis_hastings(x_train, x_valid, x_test, y_train, y_valid, y_test, q_mean):
    '''
    using the metropolis hastings MCMC sampling, randomly sample weights from a 
        gaussian proposal distribution and then run the testing set
    '''
    S = 100
    q_vars = [1, 2, 5]

    var = 1
    x_train = make_x_matrix(x_train)
    x_valid = make_x_matrix(x_valid)
    x_test = make_x_matrix(x_test)
    z = np.zeros(np.shape(q_mean))

    # perform cross validation for the sampling size S and the variance and mean of the q(w) function
    neg_lls = {}
    accuracies = {}

    for q_var in q_vars:

        predictions = np.ndarray(np.shape(y_valid))
        predictions_discrete = np.ndarray(np.shape(y_valid))

        # grab the weights and calculate the sum of r(w) values
        weights, r_sum, priors, lls, qs = hastings_sample(x_train, y_train, S, q_var, q_mean)
        for k in range(len(x_valid)):

            cur_pred = 0
            for i in range(S):
                # do the final sum
                current_r = priors[i] * lls[i] / qs[i]
                y_pred = sigmoid(np.dot(x_valid[k], weights[i]))
                # calculate the probability of prediction
                cur_pred += y_pred * current_r / r_sum
            predictions[k] = cur_pred

            if cur_pred > 0.5:
                predictions_discrete[k] = 1
            elif cur_pred < 0.5:
                predictions_discrete[k] = 0
            else:
                # if the probability of being 1 is equal to -1
                predictions_discrete[k] = -1

        # compute the negative log likelihood of the predictions and add it to the neg_lls dict
        # so that we can figure out which metrics are best
        neg_lls[q_var] = -1 * log_likelihood(predictions, y_valid)
        accuracies[q_var] = (predictions_discrete == y_valid).sum() / len(y_valid)

    print("----------- q3 -----------")
    print("Negative log-likelihoods: ")
    print(neg_lls)
    print("Accuracies: ")
    print(accuracies)
    # find the minimum of the negative log_likelihoods
    min_neg_ll = np.inf
    for v in neg_lls:
        if neg_lls[v] < min_neg_ll:
            min_neg_ll = neg_lls[v]
            min_v = v

    print("best variance q(w): " + str(min_v))

    # merge the training and validation set so we can compute the 
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    test_pred = np.ndarray(np.shape(y_test))
    test_pred_discrete = np.ndarray(np.shape(y_test))

    # compute the predictions using 100 samples (met-hastings MCMC sampling again)
    weights, r_sum, priors, lls, qs = hastings_sample(x_train, y_train, S, min_v, q_mean)

    preds_for_plot = {9: [], 10: []}
    for k in range(len(x_test)):

        cur_pred = 0
        for i in range(S):
            # do the final sum
            current_r = priors[i] * lls[i] / qs[i]
            y_pred = sigmoid(np.dot(x_test[k], weights[i])) 
            # calculate the probability of prediction
            cur_pred += y_pred * current_r / r_sum
            if k == 9 or k == 10:
                preds_for_plot[k].append(y_pred)

        test_pred[k] = cur_pred

        if cur_pred > 0.5:
            test_pred_discrete[k] = 1
        elif cur_pred < 0.5:
            test_pred_discrete[k] = 0
        else:
            # if the probability of being 1 is equal to -1
            test_pred_discrete[k] = -1

    # calculate the negative log likelihood and accuracy ratio for test set
    test_nll = -1 * log_likelihood(test_pred, y_test)
    test_acc = (test_pred_discrete == y_test).sum() / len(y_test)

    return test_nll, test_acc, min_v, preds_for_plot

def hastings_sample(x, y, sample_size, q_var, q_mean):
    '''
    generate 100 weights samples from multivariate Gaussian dist. with variance and 
        mean specified in arguments 

    param x: numpy array, x values used to make predictions
    param y: numpy array, actual y values 
    param sample_size: int, number of weight samples to take (always 100 here)
    param q_var: int, identity*var = covariance of the proposal distribution
    param q_mean: numpy array, mean of the q(w) distribution
    '''
    sample_means = []
    variance = 1
    samples = []
    # initial weight guess pulled 
    w_mean = q_mean
    w_i = proposal(q_mean, q_var)
    burned_in = False
    z = np.zeros(np.shape(q_mean))

    while len(samples) < sample_size:

        if not burned_in:
            # burn in 10000 iterations
            for j in range(1000):
                # generate uniform random variable, and random sample
                u = np.random.uniform()
                cur_w = proposal(w_mean, q_var)

                if u < min(1, (likelihood(sigmoid(np.dot(x, cur_w)), y) * gaussian(cur_w, variance, z) / likelihood(sigmoid(np.dot(x, w_i)), y) / gaussian(w_i, variance, z))):
                    # w_mean is always 1 step behind w_i
                    w_mean = w_i
                    w_i = cur_w

            burned_in = True
            samples.append(w_i)
            sample_means.append(w_mean)

        # takes w_i and w_mean from the 1000 iterations burn-in
        for j in range(100):
            # generate uniform random variable, and random sample
            u = np.random.uniform()
            cur_w = proposal(w_mean, q_var)

            if u < min(1, likelihood(sigmoid(np.dot(x, cur_w)), y) * gaussian(cur_w, variance, z) / likelihood(sigmoid(np.dot(x, w_i)), y) / gaussian(w_i, variance, z)):
                w_mean = w_i
                w_i = cur_w

        # collect sample every 100 iterations for thinning process (after 1000 burn in)
        samples.append(w_i)
        sample_means.append(w_mean)

    r_sum = 0
    priors = []
    lls = []
    qs = []
    for j in range(sample_size):
        # compute and sum r(w_js)
        pr = gaussian(samples[j], q_var, z)
        ll = likelihood(sigmoid(np.dot(x, samples[j])), y)
        q = gaussian(samples[j], q_var, q_mean)
        priors.append(pr)
        lls.append(ll)
        qs.append(q)
        r_sum += pr * ll / q

    return samples, r_sum, priors, lls, qs

def visualize_met_hast(posterior_pred):
    '''
    create the plots required for the metropolis hastings posterior and predictions values

    param posterior_pred: dict, key = flower nums, values = list of predictive posteriors
    '''
    for i in posterior_pred:
        plt.figure(i)
        plt.title("Predictive Posterior for Flower " + str(i))
        plt.xlim((0, 1))
        plt.xlabel("Pr(y*|x*, w(i))")
        plt.ylabel("Num occurences")
        plt.hist(posterior_pred[i], bins=25)
        plt.savefig("flower_" + str(i) + ".png")

if __name__ == "__main__":
    Q1a = False
    Q1b = False
    Q1c = True

    if Q1a:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data()
        x_train, x_test = np.vstack((x_train, x_valid)), x_test
        y_train, y_test = np.vstack((y_train[:,(1,)], y_valid[:,(1,)])), y_test[:,(1,)]

        log_marg_l, num_iterations, weights  = marginal_likelihoods(x_train, y_train)
        for variance in log_marg_l:
            print ("----------- sigma: " + str(variance) + " -------------")
            print("log marginal likelihood (with laplace approx): " + str(log_marg_l[variance]))
            print("number of iterations required: " + str(num_iterations[variance]))

    if Q1b:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data()
        y_train, y_valid, y_test = y_train[:,(1,)], y_valid[:,(1,)], y_test[:,(1,)]
        if not Q1a:
            junk, junk, weights  = marginal_likelihoods(x_train, y_train, variances=[1])
        
        y_train = np.asarray(y_train, int)
        y_valid = np.asarray(y_valid, int)
        y_test = np.asarray(y_test, int)
        testnll, testacc, best_var, posterior, ws_for_plot = importance_sampling(x_train, x_valid, x_test, y_train, y_valid, y_test, q_mean=weights[1])
        print("--------------------------------------")
        print("test neg LL: " + str(testnll))
        print("test acc: " + str(testacc))
        visualization(weights[1], best_var, posterior, ws_for_plot)

    if Q1c: 
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data()
        y_train, y_valid, y_test = y_train[:,(1,)], y_valid[:,(1,)], y_test[:,(1,)]

        y_train = np.asarray(y_train, int)
        y_valid = np.asarray(y_valid, int)
        y_test = np.asarray(y_test, int)

        if not Q1a:
            junk, junk, weights  = marginal_likelihoods(x_train, y_train, variances=[1])

        testnll, testacc, best_var, posterior_pred = metropolis_hastings(x_train, x_valid, x_test, y_train, y_valid, y_test, weights[1])
        print("--------------------------------------")
        print("test neg LL: " + str(testnll))
        print("test acc: " + str(testacc))
        visualize_met_hast(posterior_pred)
