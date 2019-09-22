README.md for ROB313 Assignment 2 - GLMs
Author: Mackenzie Clark
Date: 02/27/2019

The file containing all of the code for this assignment is called
a2.py and is stored in the same directory as this file and the report.
The main portion to be looked at when executing this file is the 
__main__() block at the bottom of the code. The first variables in 
the main execution block are q1, q2, and q3. These are booleans that, 
when set to true, run the code for their respective questions in the 
A2 handout.

The code uses the numpy, math, and matplotlib libraries, which are
all standard Python libraries, and it runs on Python 3.7. The datasets
are imported using the load_dataset () function provided in the 
data_utils.py file, so this must be in the location ./data/data_utis.py

Currently, the only part of the assignment that can be set up to
handle more/different datasets is the 3rd question. This would be 
done by editing the global variable at the top of the file called 
possible_datasets. In the main block, we would still have to ensure
that the new dataset gets run correctly as a regression or
classification problem.