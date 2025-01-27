# Module 1 Assignment
# part 1
'''
Write your own step function from scratch
Use the Numpy library only, if you need to use a library

Name the function step()
The function should take 1 input variable

The function should output 1, if the input value is greater than 0, and 0 otherwise

There is nothing to print in this part
'''
# WRITE YOUR CODE HERE
import numpy as numpy

# FUNCTION - step - 1 if greater than 0 otherwise 0
def step(x):
  return 1 if x > 0 else 0

'''
#1-0-0
Use your step function with 10 as input and print:
"Step function of 10 is __" - Replace the blank with the step function output
'''

# WRITE YOUR CODE HERE
output_1_0_0 = step(10)

# print output
print("Step function of 10 is", output_1_0_0)

'''
#1-0-1
Use your step function with -5 as input and print:
"Step function of -5 is __" - Replace the blank with the step function output
'''

# WRITE YOUR CODE HERE
output_1_0_1 = step(-5)

# print output
print("Step function of 10 is", output_1_0_1)

'''
#1-0-2
Then, use your step function with 0 as input print:
"Step function of 0 is __" - Replace the blank with the step function output
'''
# WRITE YOUR CODE HERE
output_1_0_2 = step(0)

# print output
print("Step function of 10 is", output_1_0_2)

# part 2
''' 
Assignment1-1: DL Regressor
Step 0: Load Libraries
'''
# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import tensorflow as tf

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
# FREEZE CODE END

'''
Step 1: Load & Prep Dataframes
We will use the 'wine quality with category value.xlsx' data file available in the file tree to the left
Load the dataset as a pandas dataframe
Explore the data dictionary using the Reference link in the right-side panel
We will build a model to predict "quality" as a numeric target

Create the x dataframe using everything except for "quality"
**Q1-1-0 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) of the x dataframe 

Create the y target using the "quality" column
**Q1-1-1 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) of the y target
'''
# WRITE YOUR CODE HERE
# input data file 
DF = pd.read_excel('wine quality with category value.xlsx', sheet_name = 0)

# generate x dataframe without quality column
x = DF.drop(columns=['quality'])

# print 16th, 17th, and 18th record (index 15 through 17) of x
print(x.iloc[15:18])

# generate y target from just quality column
y = DF['quality']

# print 16th, 17th, and 18th record (index 15 through 17) of your
print(y.iloc[15:18])

'''
Step 2: Prep train vs. test sets
Use train_test_split from sklearn to split the x dataframe and y target from the previous step
50% training and 50% for testing
Use the random seed 1693 in the train_test_split statement
** Q1-1-2 Print** the first record in the y test set
'''

# WRITE YOUR CODE HERE
# split x and y DFs 50% train, 50% test, random = 1693 (go tribe!)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5,
                                                    random_state = 1693)

# print first record of y_test
print(y_test.iloc[0])
print(x_train.shape[1])

'''
Step 3: Train Model
Create a Sequential model and name it "WineQuality"
Add one Dense hidden layer with 
(1) 7 nodes 
(2) an adequate number for input dimension, 
(3) the relu activation function, and
(4) layer name "HL1"
Add another Dense hidden layer with 
(1) 5 nodes 
(3) the relu activation function, and
(4) layer name "HL2"
Add one Dense output layer with 
(1) an adequate number of nodes
(2) an adequate activation function, and
(3) layer name "OL"
**Q1-1-3: Print** a summary of your model using model.summary()
'''
# WRITE YOUR CODE HERE
WineQuality = Sequential(name = 'WineQuality')

# LAYER 1 - Dense, 7 nodes, 12 input (13 total - 1 for quality), relu, name HL1
WineQuality.add(Dense(7, input_dim = 12, activation = 'relu', name = 'HL1'))

# LAYER 2 - Dense, 5 nodes, relu, name HL2
WineQuality.add(Dense(5, activation = 'relu', name = 'HL2'))

# LAYER 3 (output) - Dense, 5 nodes, linear for continuous, name OL
WineQuality.add(Dense(1, activation = 'linear', name = 'OL'))

# print summary
WineQuality.summary()

'''
Compile the model with 
(1) the mse as the loss function,
(2) adam as the optimizer,
(3) mse as the metrics

Train the model with:
(1) the train set from above,
(2) the train set split with 25% for validation,
(3) 150 epochs
(4) no prinout of each epoch result
There is nothing to print for this step
'''
# WRITE YOUR CODE HERE
# COMPILE - mse loss, optimizer adam, mse metric
WineQuality.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

# FIT - 25% train split for validation, 150 epochs, dont print each epoch
trainedModel = WineQuality.fit( x_train, y_train, validation_split = 0.25,
                                epochs = 150, verbose = 0)


'''
Step 4: Evaluate Model Performance
**Q1-1-4 Print** the predicted values for the first two records of the entire dataset
**Q1-1-5 Print** the model mse value from the last epoch
'''
# WRITE YOUR CODE HERE
# generate first 2 record prediction and mse from last epoch
predictedValues = WineQuality.predict(x.iloc[:2])
lastMSE = trainedModel.history['mse'][-1]

# print output
print(predictedValues)
print(lastMSE)