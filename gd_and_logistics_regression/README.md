README.md for ROB313 Assignment 3 - Gradient Descent
Author: Mackenzie Clark
Date: 03/22/2019

The file containing all of the code for this assignment is called
a3.py and is stored in the same directory as this file and the report.
The main portion to be looked at when executing this file is the 
__main__() block at the bottom of the code. The first variables in 
the main execution block are Q1 and Q2. These are booleans that, 
when set to true, run the code for their respective questions in the 
A2 handout.

The code uses the numpy, math, and matplotlib libraries, which are
all standard Python libraries, and it runs on Python 3.7. The datasets
are imported using the load_dataset () function provided in the 
data_utils.py file, so this must be in the location ./data/data_utis.py

For linear regression models (Q1) (only pumadyn32nm implemented thus 
far), the main functions used to train models are:
1. grad_desc (): this function works for full-batch and mini-batch 1 
	gradient descent algorithms by changing the parameter gd_type.
	This functions contains the implementation of gradient descent
	and returns the validation loss values to be plotted, as well
	as the optimal RMSE and learning rate
2. plot_losses (): this function takes in the RMSE losses returned from
	grad_desc and creates a figure plotting the losses vs. iteration
	number, and saves the file. It works for both GD and SGD using
	the title parameter to indicate filename and plot title. It also
	calculates and plots the linear regression RMSE as a benchmark
	to compare the gradient descent methods to.

For logistic regression models (Q2) (only iris implemented thus far),
the main functions used to train models are:
1. log_grad_desc (): this function has very similar functionality to 
	grad_desc but instead uses the log-likelihood and accuracy 
	metrics and returns both results. The function also returns a 
	dict of negative log-likehlihoods (not normalized) to be plotted.
2. plot_logistic_losses (): this function takes in the dict of 
	negative log-likelihoods from the log_grad_desc function and plots 
	them all vs. iteration number.
3. sigmoid (): helper function to compute vectorized sigmoid.
4. log-likelihood (): helper function to compute vectorized
	log-likelihood scalar values.