# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:37:16 2019

@author: ni.martynov
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# X_train=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
# y_train=np.array((0, 1, 1, 0), dtype=float)
# =============================================================================
# =============================================================================
# X = np.linspace(start=-1, stop=1, num=1000)
# noise = np.random.normal(scale = 0.1, size = (1000, ))
# target = np.power(X, 3) + noise
# X = X[:, np.newaxis]
# X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.5, \
#                                                      random_state=0)
# =============================================================================
#path_to_data = 'C:/Users/ni.martynov/Desktop/data_to_feed_alg_filled.csv'
#path_to_data_elite = 'C:/Users/ni.martynov/Desktop/elite.csv'
#path_to_data_non_elite = 'C:/Users/ni.martynov/Desktop/others.csv'
#path_to_data = 'C:/Users/ni.martynov/Desktop/nikita.csv'
path_to_data = 'C:/Users/ni.martynov/Desktop/econom_vt.csv'

data = pd.read_csv(path_to_data)
data = data[(data['Цена квартиры в рублях'] > np.quantile(data['Цена квартиры в рублях'], q = 0.01)) & \
            (data['Цена квартиры в рублях'] < np.quantile(data['Цена квартиры в рублях'], q = 0.99))]
X, target = data.drop(['Цена квартиры в рублях'], axis=1).values, data['Цена квартиры в рублях'].values
X = StandardScaler().fit_transform(X)
std = np.std(target)
target = target / std
scale = max(target)
target = target / scale
scale *= std

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, \
                                                    random_state=0)
# =============================================================================
# elite = pd.read_csv(path_to_data_elite)
# other = pd.read_csv(path_to_data_non_elite)
# elite = elite[elite['Цена'] < np.quantile(elite['Цена'], q = 0.99)]
# other = other[other['Цена'] > np.quantile(other['Цена'], q = 0.01)]
# X, target = elite.drop(['Цена'], axis=1).values, elite['Цена'].values
# scale = max(target)
# target = target / scale
# elite_train, elite_test, y_train, y_test = train_test_split(X, target, test_size=0.33, \
#                                                      random_state=0)
# other, target = other.drop(['Цена'], axis=1).values, other['Цена'].values
# target = target / scale
# X_train = np.concatenate((other, elite_train), axis=0)
# y_train = np.concatenate((target, y_train), axis=0)
# =============================================================================



def genBatch(X, target, batch_size):
    for i in range(len(X)//batch_size):
        yield X[i*batch_size:(i+1)*batch_size], \
            target[i*batch_size:(i+1)*batch_size]
    
config = {
          'input_shape': X_train.shape,
          'num_of_neurons': [1024, 512, 256, 128, 64, 32, 16],
          'layers_init': ['xavier'] * 8,
          'layers_activations': ['sigmoid', 'sigmoid', 'lrelu', 'sigmoid',\
                                 'lrelu', 'sigmoid', 'sigmoid', 'linear'],
          'bias_init': ['zero'] * 8,
          'use_l2_regularization': False,
          'l2_alpha': 1e-02,  
          'use_l1_regularization': False,
          'l1_beta': 5,
          'use_bias': True,
          'batch_size': 256,
          'num_of_epochs': 10,
          'learning_rate': 1,
          'learning_rate_bn': 1e-03,
          'clip_gradients_by_value': False,
          'max_value_to_clip_gradients': 1.,
          'min_value_to_clip_gradients': -1.,
          'batchnorm_layers': [False] * 8,
          'batchnorm_gamma_init': 'ones',
          'batchnorm_beta_init': 'zero',
          'num_of_blocks': 1,
          'dropout_layers': [False] * 7,
          'keep_prob': 0.5,
          'use_skip_connect': False
          }

# =============================================================================
# config = {
#           'input_shape': X_train.shape,
#           'num_of_neurons': [4],
#           'layers_init': ['xavier']*2,
#           'layers_activations': ['sigmoid',  'linear'],
#           'bias_init': ['zero']*2,
#           'use_l2_regularization': False,
#           'l2_alpha': 1,  
#           'use_l1_regularization': False,
#           'l1_beta': 1,
#           'use_bias': True,
#           'batch_size': 10,
#           'num_of_epochs': 100,
#           'learning_rate': 1.,
#           'learning_rate_bn': 1e-05,
#           'clip_gradients_by_value': False,
#           'max_value_to_clip_gradients': 1.,
#           'min_value_to_clip_gradients': -1.,
#           'batchnorm_layers': [False, False],
#           'batchnorm_gamma_init': 'ones',
#           'batchnorm_beta_init': 'zero',
#           'num_of_blocks': 1
#           }
# =============================================================================


