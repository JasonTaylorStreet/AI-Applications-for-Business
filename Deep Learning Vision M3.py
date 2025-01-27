# M3 Assignment
# Part 1
'''
M3 Assignment-0
1. Write the Tanh function from scratch using only the Numpy library. You are NOT allowed to call a pre-existing tanh() function from a public Python library.
2. Use a for loop to call the Tanh function you wrote above using input values ranging from -10 to 10 (including both -10 and 10) with increments of 5
3. For each function call, print the output as "The Tanh value of x is y" in a new line.
   Replace x and y with appropriate values.
   For example, the firs line should be "The Tanh value of -10 is -0.9999999958776926"
   Copy and paste quoted texts to minimize the risk of typing errors that are very hard for you to debug.
'''

# WRITE YOUR CODE HERE
import numpy as np
# FUNCTION - Tanh from scratch using only Numpy
def tanhF(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Loop -10 through 10 every 5 with tanhF and print each
for x in range(-10, 11, 5):
  y = tanhF(x)
  print("The Tanh value of", x, "is", y)


# Part 2
''' 
M3 Assignment-1
Fashion MNIST Recognition
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

'''
Step 0: Load libraries
There is nothing to print during this step.
'''

# WRITE YOUR CODE HERE
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

''' 
Step 1: Load & prep dataframe 
Read dataset from Fashion MNIST (NOT MNIST).
Tip: Follow Keras documentation.

**Q3-1-0 Print "0. the 16th image in the X train set: xxx" - Replace xxx with the proper value(s) using the object[x] format.
**Q3-1-1 Print "1. the Y label of the 16th item in the original train set: xxx" - Replace xxx with the proper value(s) using the object[x] format.
**Q3-1-2 Print "2. the actual description of the 16th item's label: xxx" - Replace xxx with the proper value(s) (e..g, 'Bag')
Tip: Visit the documentation site for the data dictionary which provides actual descriptions of Y labels.
Tip: You can inspect the image in Colab to make sure the class label is correct ;)

'''
# WRITE YOUR CODE HERE
# LOAD - fashion mnist data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print 16th image of x_train
print("0. the 16th image in the X train set:\n", x_train[15])

# print 16th y label of y_train
print("1. the Y label of the 16th item in the original train set:", y_train[15])

# assign description for each numeric label then print 16th label description
# label descriptions from https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file#get-the-data
label_descrip = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
print("2. the actual description of the 16th item's label:", label_descrip[y_train[15]])

'''
Step 2: Prep Train vs. Test Sets
Scale X values in both train and test sets to the range of 0 ~ 1.
**Q3-1-3 Print "3. the scaled X values of the 16th item in the train set: xxx" - Replace xxx with the proper value(s) using the object[x] format.
One-hot code train and test labels (i.e., Y values) using to_categorical.
**Q3-1-4 Print "4. the one-hot coded Y label of the 16th item in the train set: xxx" - Replace xxx with the proper value(s) using the object[x] format.
'''

# WRITE YOUR CODE HERE
# SCALE - X 0 to 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# print train 16th image scaled
print("3. the scaled X values of the 16th item in the train set:\n", x_train[15])

# ENCODE - one-hot y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print train 16th label one-hot encoded
print("4. the one-hot coded Y label of the 16th item in the train set:\n", y_train[15])

'''
Step 3: Train Model
Create a Sequential model.
Add a layer to flatten the input image set.
Add a dense hidden layer with 5 nodes and relu as the activation. 
Add a dropout layer with 15% dropout rate.
Add a second dense hidden layer with 4 nodes and the tanh activation function.
Add a dense output layer with the appropriate activation function and the appropriate output dimension.
Do NOT name the model or any of the layers.

**Q3-1-5 Print a summery of your model.

Compile the model with
- the adam optimizer
- categorical_crossentropy as loss function
- accuracy as the metric
Train the model with 
- the train set that you loaded from Keras
- 5 epoches
- batch size of 2000
Tip: Turn off the printing of epoch-by-epoch output so it's easier to inspect your printouts.
'''
# WRITE YOUR CODE HERE
# no name Sequential model
model = models.Sequential()

# LAYER 1 - flatten
model.add(layers.Flatten(input_shape = (28, 28)))

# LAYER 2 - dense, 5 nodes, relu activation
model.add(layers.Dense(5, activation = 'relu'))

# LAYER 3 - 15% dropout
model.add(layers.Dropout(0.15))

# LAYER 4 - dense, 4 nodes, tanh activation
model.add(layers.Dense(4, activation = 'tanh'))

# LAYER 5 (output) - dense, 10 nodes(dimension), softmax activation
model.add(layers.Dense(10, activation = 'softmax'))

# print model summary
model.summary()

# COMPLILE - categorical_crossentropy loss, adam optimizer, accuracy metrics
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# FIT - train on train, 5 epochs, batch size 2000, dont print each epoch
fit_model = model.fit(x_train, y_train, epochs = 5, batch_size = 2000, verbose = 0)

'''
Step 4: Evaluate model performance
**Q3-1-6 Print "6. train accuracy rate for the 4th epoch: xxx" - Replace xxx with the proper value(s) : xxx" - Replace xxx with the proper value(s).
Use the model you trained to make a prediction on the 16th item in the train set.
Tip: Use the object[x:y] format to select one record for making predictions.
**Q3-1-7 Print "7. the predicted class label for the 16th item in the train set: xxx" - Replace xxx with the proper value(s). This should be an actual class label (e.g., 'Bag').
'''

# WRITE YOUR CODE HERE
# EVALUATE
# print 4th epcch accuracy (train)
print("6. train accuracy rate for the 4th epoch:", fit_model.history['accuracy'][3])

# PREDICT - train 16th item
predict16 = model.predict(x_train[15:16])

# print label of prediction
predict16_label = label_descrip[np.argmax(predict16)]
print("7. the predicted class label for the 16th item in the train set:", predict16_label)