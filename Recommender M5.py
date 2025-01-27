# Module 5 Assignment
# Part 1
''' 
M5 Assignment-0: Create and use a subclass of the keras.layers.Layer class
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
# create subclass of keras.layers.Layer
class MinMaxLayer(tf.keras.layers.Layer):
  def __init__(self, units = 12, input_dim = 6):
    super(MinMaxLayer, self).__init__()
    self.units = units
    self.input_dim = input_dim
  
  def call(self, inputs):
    # verify length of inputs, else error statement
    if len(inputs) != self.input_dim:
      return f"Error when checking input: expected to have {self.input_dim} as input_dim, but got array with {len(inputs)} elements"
    
    # MinMax scaling without sklearn
    min_val = tf.math.reduce_min(inputs)
    max_val = tf.math.reduce_max(inputs)
    scaled_output = (inputs - min_val) / (max_val - min_val)
    return tf.convert_to_tensor(scaled_output, dtype = tf.float64)

y1 = np.array([2,4,6,8,10])
y2 = np.array([5,10,15])

# WRITE YOUR CODE HERE
# First layer using y1 and correct number of input_dim
MyFirstLayer1 = MinMaxLayer(input_dim = 5)
result_y1 = MyFirstLayer1(y1)
if isinstance(result_y1, str):
  print(result_y1)
else:
  print("MyFirstLayer1's output is", result_y1)

# second layer using y2 and default input_dim
MySecondLayer2 = MinMaxLayer()
result_y2 = MySecondLayer2(y2)
if isinstance(result_y2, str):
  print(result_y2)
else:
  print(f"MySecondLayer2's output is", result_y2)


# Part 2
'''
M5 Assignment-1: Build an embedding model
'''
# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress warnings

import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import pprint

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
# Import additional lirbaries/modules as needed
# FREEZE CODE END

# WRITE YOUR CODE HERE
# load movielens/latest-small-movies dataset
train_data = tfds.load('movielens/latest-small-movies', split = 'train')

# print first 2 records with pprint
for x in train_data.take(2).as_numpy_iterator():
  pprint.pprint(x)

# FUNCTION pull movie titles
def pull_titles(x):
  return x['movie_title']

# string lookup layer for vocab
movie_title_lu = tf.keras.layers.StringLookup(mask_token = None)
movie_title_data = train_data.map(pull_titles)
movie_title_lu.adapt(movie_title_data)

# print vocabulary size
vocab_size = movie_title_lu.vocabulary_size()
print(vocab_size)

# print 30 & 31 pull_titles
titles_3031 = movie_title_lu.get_vocabulary()[29:31]
print(titles_3031)

# print raw tokens of 30 & 31
print(movie_title_lu(titles_3031))

# define keras embedding object
embed_layer = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = 28)

# combine lookup and embedding
embed_model = tf.keras.Sequential([movie_title_lu, embed_layer])

# print embedding of star wars 1977
sw_embed = embed_model(tf.constant(["Star Wars (1977)"]))
print(sw_embed)


# Part 3
''' 
M5 Assignment-2: Create and inspect a recommender system
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
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from typing import Dict, Text

# load movielens/latest-small-ratings & movielens/latest-small-movies train datasets 
ratings = tfds.load('movielens/latest-small-ratings', split="train")
movies = tfds.load('movielens/latest-small-movies', split="train")

# select movie title from both, user id from ratings
def transform_ratings(x):
    return {"movie_title": x["movie_title"], "user_id": x["user_id"]}
ratings = ratings.map(transform_ratings)

def transform_movies(x):
    return x["movie_title"]
movies = movies.map(transform_movies)

# M5 lab model
class MovieLensModel(tfrs.Model):
    def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)

# vocab for user IDs
uID_vocab = tf.keras.layers.StringLookup(mask_token=None)
def extract_uID(x):
    return x["user_id"]
uID_vocab.adapt(ratings.map(extract_uID))

# print user IDs vocab size
print(uID_vocab.vocabulary_size())

# print IDs 30 & 31
uID_3031 = uID_vocab.get_vocabulary()[29:31]
print(uID_3031)

# vocab for movie titles
mTitle_vocab = tf.keras.layers.StringLookup(mask_token = None)
def extract_mTitle(x):
    return x
mTitle_vocab.adapt(movies.map(extract_mTitle))

# print movie title vocab size
print(mTitle_vocab.vocabulary_size())

# print titles 30 & 31
mtitles_3031 = mTitle_vocab.get_vocabulary()[29:31]
print(mtitles_3031)

# define sequential model using uID_vocab and output 28
user_model = tf.keras.Sequential([uID_vocab, tf.keras.layers.Embedding(uID_vocab.vocabulary_size(), 28)])

# print user id 17 embeddings
user_embed = user_model(tf.constant(["17"]))
print(user_embed)

#define sequential model using mTitle_vocab and output 28
movie_model = tf.keras.Sequential([mTitle_vocab, tf.keras.layers.Embedding(mTitle_vocab.vocabulary_size(), 28)])

# print star wars 1977 embedding
movie_embed = movie_model(tf.constant(["Star Wars (1977)"]))
print(movie_embed)

# task movie batches of 256, apply movie model each batch
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(movies.batch(256).map(movie_model)))

# create MovieLensModel w/ user & movie models and task
model = MovieLensModel(user_model=user_model, movie_model=movie_model, task=task)

# COMPILE - SGD optimizer, learning rate 0.75
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.75))

# TRAIN - w/ ratings, batch size 3500, 3 epochs
model.fit(ratings.batch(3500), epochs=3, verbose=0)

# Recmnd model w/ factorizedtopk streaming
index = tfrs.layers.factorized_top_k.Streaming(model.user_model)

# index movies, batch of 150 movies (lambda to gen tuple)
index = index.index_from_dataset(movies.batch(150).map(lambda title: (title, model.movie_model(title))))

# make top 5 recommendations for user 17
_, top_titles = index(tf.constant(["17"]))

# print the recommendations
print("Top 5 recommendations for user 17:", top_titles[0, :5].numpy())
