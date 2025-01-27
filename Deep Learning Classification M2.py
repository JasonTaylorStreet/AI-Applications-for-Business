# M2 Assignment
# Part 1
'''
1. Write the Softmax function from scratch using only the Numpy library.
2. Do NOT call a built-in softmax function from any Python library.
3. You have to write the Softmax function from scratch on your own.
4. Your Softmax function should take a vector as input, and output a vector of Softmax values.
5. Call your Softmax function using [-1,3.8,2.7,4.5] as input.
6. **Q2-0-0 Print** the output of the function call as "The Softmax values of [-1,3.8,2.7,4.5] are [x x x x]"
   Replace [x x x x] with appropriate values.
7. Call the same Softmax function you created using [-12.5,5,3,23.4, -.09] as input.
6. **Q2-0-1 Print** the output of the function call as "The Softmax values of [-12.5,5,3,23.4, -.09] are [x x x x x]"
   Replace [x x x x x] with appropriate values.
'''

# WRITE YOUR CODE HERE
import numpy as np
# FUNCTION - softmax from scratch
def softmaxF(vector):
  e = np.exp(vector)
  return e / np.sum(e)

# input data
input_1 = [-1,3.8,2.7,4.5]
input_2 = [-12.5,5,3,23.4, -.09]

# print output
print("The Softmax values of [-1,3.8,2.7,4.5] are", softmaxF(input_1))
print("The Softmax values of [-12.5,5,3,23.4, -.09] are", softmaxF(input_2))

# Part 2
'''
Step 0: Load Libraries
The starter code given below only imports some, not all, of the libraries you need.
You should load the other libraries you need below.
'''
# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress warnings

import random
import numpy as np
import tensorflow as tf

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
# FREEZE CODE END

# WRITE YOUR CODE HERE
# additional libraries
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split

'''
Step 1: Load & Prep Dataframes
We will use the 'wine quality combined.xlsx' data file available in the file tree to the left.
Load the dataset as a pandas dataframe.
Explore the data dictionary using the Reference link in the right-side panel.
We will build a model to predict "quality" as a categorical target.

One-hot code categorical variables within the dataframe using the get_dummies() function from Pandas.
Reference the Pandas documentation to see how to use get_dummies() properly.
Treat "quality" and "category" as categorical variables (meaning the numbers indicate categories, even though the variables are integers.)
Note that you can decide when to one-hot code each of the variables,
and whether to do so together or separately,
as long as your printouts below are correct.

Create the x dataframe using everything except for "quality."
**Q2-1-0 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) of the x dataframe after one-hot coding from the previous step.

Create the y target using the one-hot coded "quality" columns.
**Q2-1-1 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) from the one-hot coded y target
'''

# WRITE YOUR CODE HERE
# load data Note: Variable table != header row. color vs category
DF = pd.read_excel('wine quality combined.xlsx')

# one-hot encoded category
DF = pd.get_dummies(DF, columns = ['category'])

# create X dataframe with all columns EXCEPT quality
x = DF.drop('quality', axis = 1)

# create y target and ensure one-hot encoded
y = pd.get_dummies(DF['quality'].astype(str), prefix = 'quality')

# print output
print(x.iloc[15:18])
print(y.iloc[15:18])

'''
Step 2: Prep train vs. test sets
Use train_test_split from sklearn to split the x dataframe and y target from the previous step (after one-hot coding).
50% training and 50% for testing
Use the random seed 1693 in the train_test_split statement.
** Q2-1-3 Print** the first record in the y test set
** Tip: Python uses zero-based, not one-based, indexing. **
'''

# WRITE YOUR CODE HERE
# split x and y, 50% test and 50% train, random = 1693 (go tribe!)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 1693)

# print 1st record of y_test
print(y_test.iloc[0])

'''
Step 3: Train model
Create a Sequential model with no name
Add one Dense hidden layer with
(1) 15 nodes,
(2) an adequate number for input dimension,
(3) the tanh activation function, and
(4) no layer name
Add a second Dense hidden layer with
(1) 20 nodes,
(3) the relu activation function, and
(4) no layer name
Add one Dense output layer with
(1) an adequate number of nodes
(2) the appropriate activation function, and
(3) no layer name
**Q2-1-3 Print** a summary of your model using model.summary()
'''

# WRITE YOUR CODE HERE
# no name sequential model
model = Sequential()

# LAYER 1 - Dense, 15 nodes, input=shape, tanh activation
model.add(Dense(15, input_dim = x_train.shape[1], activation = 'tanh'))

#LAYER 2 - Dense, 20 nodes, relu activation, no name
model.add(Dense(20, activation = 'relu'))

# LAYER 3 (output) - Dense, nodes=shape, softmax activation, no name
model.add(Dense(y_train.shape[1], activation = 'softmax'))

# print summary
model.summary()

'''
Compile the model with
(1) CategoricalCrossentropy as the loss function,
(2) sgd as the optimizer
(3) accuracy as the metric

Fit the model with
(1) the train set you created in Step 2, and do NOT use the test set you created in Step 2 as validation_data.
(3) 100 epochs
(4) no prinout of each epoch

Do not specify learning rate, momentum, or other parameters that are not required in the instructions.

There's nothing to print for this part
'''

# WRITE YOUR CODE HERE
# COMPILE - categorical_crossentropy loss, sgd optimizer, accuracy metrics
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# FIT - no validation, 100 epochs, dont print each epoch
model.fit(x_train, y_train, epochs = 100, verbose = 0)

'''
Step 4: Evaluate Model Performance
**Q2-1-4 Print** test accuracy using the evaluate() function from Keras, and the test set you created in step 2.
         Format the printout as 'test_acc: .5'
         Replace .5 with the correct value
Convert predicted probabilities into predicted classes
CAUTION: Think carefully about how get_dummies() works and how you should structure the conversion so you predict the correct class labels.
**Q2-1-5 Print** predicted quality category (i.e. quality labels) for the 16ths, 17th, and 18th records from the original dataset.
**Q2-1-6 Calculate and print** the model's accuracy rate using the entire original dataset.
'''

# WRITE YOUR CODE HERE
# evaluate and print test_acc
test_loss, test_metric = model.evaluate(x_test, y_test, verbose = 0)
print("test_acc:",test_metric)

# predict probabilites
predict_probs = model.predict(x)

# convert into predicted class (still one-hot encoded)
predict_class_one_hot = pd.DataFrame(predict_probs, columns = y.columns).idxmax(axis = 1)

# convert one-hot class into categorical (orignially integer)
predict_class = predict_class_one_hot.str.replace('quality_', '').astype(int)

# print quality labels for 16-18 record
print(predict_class.iloc[15:18].tolist())

# calculate model accuracy
total_predictions = len(y)
sum_correct_predicts = (predict_class == y.idxmax(axis = 1).str.replace('quality_', '').astype(int)).sum()
model_accuracy = sum_correct_predicts / total_predictions

# print output
print(model_accuracy)