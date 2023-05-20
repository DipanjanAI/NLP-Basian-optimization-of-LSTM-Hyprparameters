
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
space = {
        'num_of_units': hp.quniform('num_of_units',10,100,10),
        'learning_rate' : hp.quniform ('learning_rate', 0.0001, 0.01,0.001),
        # 'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])
    }

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense
from keras.models import Model

import os
import numpy as np
from keras.datasets import imdb

def objective(space):
    
    num_of_units=int(space['num_of_units'])
    learning_rate1=space['learning_rate']
    
    
    
    
    """defining the data parameters"""
    
    
    """num_words sets the maximum number of unique words to be included in the vocabulary, which is used to map words to integers. In this case, the top 88,584 most frequently occurring words in the dataset will be selected."""
    num_words = 88584
    """This line loads the IMDB movie review dataset and splits it into training and test sets. The imdb.load_data() function returns a tuple of two lists, where the first list contains the reviews as a sequence of word indices and the second list contains the corresponding sentiment labels (0 for negative and 1 for positive). By setting num_words to num_words, only the top num_words most frequently occurring words are retained in the dataset."""
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = num_words)
    
    
    
    
    
    """
    This is prepocessing to trim the length of the sentenses to suible sizes
    
    The `pad_sequences()` function is used to ensure that all the sequences (i.e., sentences or input data) have the same length. This is important for training neural networks since they require a fixed input size. 
    
    the `pad_sequences()` function is used to pad the sequences in the `train_data` and `test_data` datasets to a maximum length of `max_length`. This ensures that all the sequences in these datasets have the same length, which is necessary to feed them into the neural network model for training and testing.
    
    The `max_length` parameter specifies the maximum length of the sequences. Any sequences that are shorter than this length are padded with zeros at the end, and any sequences that are longer than this length are truncated.
    
    After the padding has been applied, the modified sequences are assigned back to the original variables `train_data` and `test_data`, respectively. The modified sequences can then be fed into the neural network for training and testing.
    
    """
    
    
    
    
    max_length= 250
    sample_length = 64
    import keras
    import tensorflow as tf
    from keras.utils import pad_sequences
    
    train_data=pad_sequences(train_data,max_length)
    test_data=pad_sequences(test_data,max_length)
    
    
    
    
    """
    model architecture and optimization
    
    
    This code defines a Sequential model in TensorFlow Keras that can be used for binary classification tasks.
    
    Here's a breakdown of what each line does:
    
    model = tf.keras.Sequential([]): This creates a new instance of the Sequential class in TensorFlow Keras, which allows us to stack layers on top of each other to create a neural network.
    
    tf.keras.layers.Embedding(num_words, num_of_units): This is the first layer in the model. It is an Embedding layer, which takes an integer input (representing the index of a word in a vocabulary) and converts it to a dense vector of fixed size (in this case, num_of_units). The num_words argument specifies the size of the vocabulary (i.e., the maximum integer index that can be used as input).
    
    tf.keras.layers.LSTM(num_of_units): This is the second layer in the model. It is a LSTM layer, which stands for Long Short-Term Memory. LSTM layers are commonly used for processing sequences of data (e.g., text or time-series data). 
    
    tf.keras.layers.Dense(1, activation='sigmoid'): This is the final layer in the model. It is a Dense layer with a single unit, which makes it suitable for binary classification tasks. The sigmoid activation function is used to ensure that the output of the layer is a probability between 0 and 1.
    
    In summary, this model takes integer inputs (representing words in a vocabulary) and converts them to dense vectors using an Embedding layer. The resulting vectors are then processed by an LSTM layer to capture the sequence information, and finally passed through a Dense layer with a sigmoid activation function to produce a binary classification output.
    """
    
    model=tf.keras.Sequential([
        tf.keras.layers.Embedding(num_words,num_of_units),
        tf.keras.layers.LSTM(num_of_units),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    
    
    from keras import optimizers
    sgd = optimizers.RMSprop(learning_rate=learning_rate1)
    model.compile(loss="binary_crossentropy",optimizer=sgd,metrics=['accuracy'])
    history=model.fit(train_data,train_labels,epochs=20
                      ,validation_split=0.2)
    
    
    
    results=model.evaluate(test_data,test_labels)
    print(results)
    
    return -results[1]
    


from sklearn.model_selection import cross_val_score
trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 80,
            trials= trials)



crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200,5:1300,6:1500}

print(crit[best['criterion']])
print(feat[best['max_features']])
print(est[best['n_estimators']])
