# Module 6 Assignment
'''
M6 Assignment: Reinforcement Learning
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
# Import additional lirbaries/modules as needed
# FREEZE CODE END

# WRITE YOUR CODE HERE
# generate arms array of 2 arms with [mean, std dev.]
arms = np.array([[3, 1], [6, 2]])

# print arms array
print("0:", arms)

# FUNCTION - randsom reward based on arms mean & std dev.
def reward(arm):
  mean, sd = arm
  return np.random.normal(mean, sd)

# print reward mean = 5, std dev. = 1
print("1:", reward([5, 1]))

# generate memory array with start of 0 for both [arm, reward]
start_arm = 0
start_reward = 0.0
memory = np.array([start_arm, start_reward]).reshape(1, 2)

# print memory array
print("2:", memory)

# FUNCTION - greedy method
def bestArm(a):
  bestArm = 0
  bestMean = 0
  for u in a:
    this_action = a[np.where(a[:, 0] == u[0])]
    avg = np.mean(this_action[:, 1])
    if bestMean < avg:
      bestMean = avg
      bestArm = u[0]
  return int(bestArm)

# run 20 trials with esilon-greedy (30% exploit, 70% explore)
num_trials = 20
eps = 0.7
for trial in range(num_trials):
  if np.random.random() > eps:
    selected_arm = bestArm(memory)
    new_record = np.array([[selected_arm, reward(arms[selected_arm])]])
    memory = np.concatenate((memory, new_record), axis = 0)
  else:
    selected_arm = np.random.randint(0, 2)
    reward_score = reward(arms[selected_arm])
    new_record = np.array([[selected_arm, reward_score]])
    memory = np.concatenate((memory, new_record), axis = 0)
  cumulative_mean = np.mean(memory[:,1])
  print("3:", cumulative_mean)

# print memory array
print(memory)