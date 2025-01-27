# Module 4 Assignment
# Part 1
'''
1. Write the entropy function for binary classification from scratch using only Python's math module. Do NOT use Numpy.
2. Do NOT call a built-in entropy function from any Python library.
3. Use the math library only. Do not use any other library.
4. You have to write the entropy function from scratch on your own. Name it MyEntropy.
5. Your entropy function should take a probability value for one of the two classes as input, and output its entropy value.
6. Make sure the frozen starter code  runs properly without error and produces correct output.
'''

import math
# WRITE YOUR CODE HERE
# FUNCTION - entropy from scratch only using math (no NumPy)
def MyEntropy(p):
  # prevent error if p = 0 or 1
  if p == 0 or p == 1:
    return 0
  else:
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

# FREEZE CODE BEGIN
output0 = MyEntropy(.2)
print("The entropy value of probability .2 is " + str(output0)) # Q4-0-0

output1 = MyEntropy(.8)
print("The entropy value of probability .8 is " + str(output1)) # Q4-0-1

output2 = MyEntropy(.5)
print("The entropy value of probability .5 is " + str(output2)) # Q4-0-2
# FREEZE CODE END

# Part 2
'''
M4 Assignment-1: RNN for Time Series Prediction.
This assignment is based on the Time Series Prediction lab 10.9.6 from ISLP Chapter 10.
Please use the textbook lab as a reference.
Note that the textbook lab is written using PyTorch.
You should write your model using Tensorflow.
The goal is to predict log_volume using lagged data.

Step 0: Load Libraries
Load the libraries you need in this lab
'''

# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF warnings

import random
import numpy as np
import tensorflow as tf
import pandas as pd

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
# FREEZE CODE END

# WRITE YOUR CODE HERE
# load needed libraries here
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# FREEZE CODE BEGIN
'''
Step 1: Load & Prep Dataframes
Load the NYSE dataset from the NYSE.csv file available in the file tree to the left.
The date column gives you the timestamps of the time Series.
The train column indicates True for records to be used in the train set, and False for those to be used in the test set.
For this step, let's keep only these 3 columns: 'DJ_return', 'log_volume', 'log_volatility'
Standardize all 3 columns using ScikitLearn's StandardScaler.
In the starter code given below:
  - cols is a list of the names of these 3 columns.
  - X is a dataframe that contains only these 3 columns from NYSE.csv.
**Q4-1-0 Print "0. the shape of datafrmae X: xxx" - Replace xxx with the proper value
**Q4-1-1 Print "1. the first record of dataframe X: xxx" - Replace xxx with proper values
'''
cols = ['DJ_return', 'log_volume', 'log_volatility']
# FREEZE CODE END

# WRITE YOUR CODE HERE
# load data
NYSE = pd.read_csv("NYSE.csv")

# use standardscalar to standardize and keep only 3 listed columns
scaler = StandardScaler(with_mean = True, with_std = True)
X = pd.DataFrame(scaler.fit_transform(NYSE[cols]), columns = cols, index = NYSE.index)

# print outputs
print("0. the shape of dataframe X:", X.shape)
print("1. the first record of dataframe X:\n", X.iloc[0])
print("1. the first record of dataframe X:\n", X.iloc[0].to_dict())

# FREEZE CODE BEGIN
'''
Use code from the textbook lab to set up lagged versions of these 3 data columns (using the starter code given to you here.)
Add column 'train' from the original dataset to the current dataframe X as the last column (to the right).
**Q4-1-2 Print "2. the shape of dataframe X with lags: xxx" - Replace xxx with the proper value
**Q4-1-3 Print "3. the first record of the data frame with lags: xxx" - Replace xxx with proper values
'''
# FREEZE CODE END

# WRITE YOUR CODE HERE
# generate lagged version of 3 selected columns
for lag in range(1, 6):
  for col in cols:
    # generate and insert new column for lagged data
    newcol = np.zeros(X.shape[0]) * np.nan
    newcol[lag:] = X[col].values[:-lag]
    X.insert(len(X.columns), "{0}_{1}".format(col, lag), newcol)

# add original train data column
X['train'] = NYSE['train']

# print outputs
print("2. the shape of dataframe X with lags:", X.shape)
print("3. the first record of the data frame with lags:\n", X.iloc[0].to_dict())

# FREEZE CODE BEGIN
'''
Drop any rows with missing values using the dropna() method.
**Q4-1-4 Print "4. the shape of dataframe X with lags: xxx" - Replace xxx with the proper value
**Q4-1-5 Print "5. the first record of dataframe X with lags: xxx" - Replace xxx with proper values
'''
# FREEZE CODE END

# WRITE YOUR CODE HERE
# drop rows with missing values
X = X.dropna()

# print outputs
print("4. the shape of dataframe X with lags:", X.shape)
print("5. the first record of dataframe X with lags:\n", X.iloc[0].to_dict())

# FREEZE CODE BEGIN
'''
Create the Y response target using the 'log_volume' column from dataframe X.
Extract the 'train' column from dataframe X as a separate variable called train. Drop the 'train' column from dataframe X.
Later on we will use the train variable to split the dataset into train vs. test.
Drop the current day’s DJ_return (the "DJ_return" column) and log_volatility from dataframe X.
- Current day refers to the non-lagged columns of these two variables. 
- In other words, remove these two X features, and also the Y response that came from dataframe X.
**Q4-1-6 Print "6. the first 3 records of the Y target : xxx" - Replace xxx with proper values.
**Q4-1-7: Print "7. the first 3 records of the train variable: xxx" - Replace xxx with proper values.
**Q4-1-8: Print "8. the first 3 records of dataframe X: xxx" - Replace xxx with proper values.
'''
# FREEZE CODE END

# WRITE YOUR CODE HERE
# make log_volume Y target, generate separate train variable, then drop from x along with DJ_return and log_volatility
Y, train = X['log_volume'], X['train']
X = X.drop(columns = ['train'] + cols)

# print outputs
print("6. the first 3 records of the Y target:\n", Y.head(3).to_dict())
print("7. the first 3 records of the train variable:\n", train.head(3).to_dict())
print("8. the first 3 records of dataframe X:\n", X.head(3).to_dict())

# FREEZE CODE BEGIN
'''
To fit the RNN, we must reshape the X dataframe, as the RNN layer will expect 5 lagged versions of each feature as indicated by the (5,3) shape of the RNN layer below. 
We first ensure the columns of our X dataframe are such that a reshaped matrix will have the variables correctly lagged. 
We use the reindex() method to do this. 
The RNN layer also expects the first row of each observation to be earliest in time..
So we must reverse the current order.
Follow the textbook lab code to reorder/reindex the columns properly.
**Q4-1-9: Print "9. the first 3 records of X after reindexing: xxx" - Replace xxx with proper values.
'''
# FREEZE CODE END

# WRITE YOUR CODE HERE
# reindex columns in lagged order
ordered_cols = []
for lag in range(5, 0, -1):
  for col in cols:
    ordered_cols.append('{0}_{1}'.format(col, lag))
X = X.reindex(columns = ordered_cols)

# print output
print("9. the first 3 records of X after reindexing:\n", X.head(3).to_dict())

'''
Reshape dataframe X as a 3-D Numpy array such that each record/row has the shape of (5,3). Each row represents a lagged version of the 3 variables in the shape of (5,3). 
**Q4-1-10: Print "10. the shape of X after reshaping: xxx" - Replace xxx with proper values.
**Q4-1-11: Print "11. the first 2 records of X after reshaping: xxx" - Replace xxx with proper values. 
''' 
# WRITE YOUR CODE HERE
# reshape as 3-D array
X_reshape = X.to_numpy().reshape((-1, 5, 3))

# print outputs
print("10. the shape of X after reshaping:", X_reshape.shape)
print("11. the first 2 records of X after reshaping:\n", X_reshape[:2])

'''
Now we are ready for RNN modeling.
Set up your X_train, X_test, Y_train, and Y_test using the X dataframe, Y response target, and the train variable you have created above.
Include records where train = True in the train set, and train = False in the test set.
Configure a Keras Sequential model with
(1) proper input shape,
(2) SimpleRNN layer with 12 hidden units, the relu activation function, and 10% dropout
(3) a proper output layer.
Do not name the model or any of the layers.
**Q4-1-12: Print a summary of your model. 
'''

# WRITE YOUR CODE HERE
# split data
X_train, X_test, Y_train, Y_test = X_reshape[train == True], X_reshape[train == False], Y[train == True], Y[train == False]

# set up model
model = Sequential()

# LAYER 1 - SimpleRNN, 12 units, relu activation, 10% dropout
model.add(layers.SimpleRNN(12, activation = 'relu', dropout = 0.1, input_shape = (5, 3)))

# LAYER 2 (output) - dense
model.add(layers.Dense(1))

# print model summary
model.summary()

'''
Compile the modle with
(1) the adam optimizer,
(2) MSE as the loss,
(3) MSE as the metric.

Fit the model with
(1) 200 epochs,
(2) batch size of 32.
No need to print epoch-by-epoch progress.

There is nothing to print for this step.
'''

# WRITE YOUR CODE HERE
# COMPILE - adam optimizer, MSE loss, MSE metrics
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

# FIT - 200 epochs, batch size 32, dont print each epoch
model.fit(X_train, Y_train, epochs = 200, batch_size = 32, verbose = 0)

'''
Evaluate the model using model.evaluate() with the test set
**Q4-1-13 Print "13. Test MSE: xxx" - Replace xxx with the proper value.
'''

# WRITE YOUR CODE HERE
# EVALUATE
test_loss, test_metrics = model.evaluate(X_test, Y_test, verbose = 0)

# print output
print("13. Test MSE:", test_metrics)