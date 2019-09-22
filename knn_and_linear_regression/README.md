author: Mackenzie Clark
date: 02/12/2019

The main files used to generate results and build k-NN models are 
the following:

1. main_knn.py:	(q1 and q2)
	The main_knn.py file contains a test () function that was initially
	created to run some small tests on the calculation functions. This
	function still remains in tact, but does not provide a complete test
	of all functions.

	The rest of the code is called simply by changing the indices of the
	2 loops in the main block that loop through the global var "possible_
	datasets". Indices 0,1,2 correspond to regression datasets, and can be
	run using the first loop (possible_datasets[0:3]). The second loop can
	be used to run all the classification sets (possible_datasets[3:5]).

	The function will return and print out the optimal k values, RMSE for
	validation and the test data RMSE. 

2. vectorized_knn.py: (q3)
	The vectorized_knn.py has 4 functions (part a, b, c, d), all the
	functions should be run together from the main block to ensure that
	the correct plots are output. This file takes approx. 30 mins to 
	execute and should only be done when necessary. 

3. svd.py: (q4)
	The svd.py file runs the singular value decomposition on each of the
	regression and classification datasets. Similarly to the main_knn.py,
	to run the SVD on the datasets (regression and classification), we 
	simply just need to change the indices in the main block to include
	the correct datasets out of the possible_datasets list.

All of the python files require Python3.7 and import the data_utils.py
file from within a directory called data (i.e. data.data_utils).