# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:21:40 2019

@author: ni.martynov

"""
import numpy as np
from activations import *
from initialization import *
from utils import *
from base import Base


class Perceptron(Base):
    def __init__(self, config):
        self._input = None
        self._input_shape = config['input_shape']
        self._num_of_neurons = config['num_of_neurons']
        self._layers_init = config['layers_init']
        self._use_bias = config['use_bias']
        self._bias_init = config['bias_init']
        self._use_l2_regularization = config['use_l2_regularization']
        self._l2_alpha = config['l2_alpha']
        self._use_l1_regularization = config['use_l1_regularization']
        self._l1_beta = config['l1_beta']
        self._layers_activations = config['layers_activations']
        self._batch_size = config['batch_size']
        self._target = None
        self._learning_rate = config['learning_rate']
        self._learning_rate_bn = config['learning_rate_bn']
        self._clip_gradients_by_value = config['clip_gradients_by_value']
        self._max_value_to_clip_gradients = \
                            config['max_value_to_clip_gradients']
        self._min_value_to_clip_gradients = \
                            config['min_value_to_clip_gradients']
        self._batchnorm_layers = config['batchnorm_layers']
        self._batchnorm_gamma_init = config['batchnorm_gamma_init']
        self._batchnorm_beta_init = config['batchnorm_beta_init']
        
        self._use_skip_connect = config['use_skip_connect']
        self._num_of_blocks = config['num_of_blocks']
        self._size_of_block = len(self._num_of_neurons)
        
        self._dropout_layers = config['dropout_layers']
        self._keep_prob = config['keep_prob']
        
    def initLayers(self):
        self._layers = []
        self._biases = []
        self._bn_gammas = {}
        self._bn_betas = {}
        
        self.InitRegLayers()
        if sum(self._batchnorm_layers):
            self.InitBnLayers()
            
        if self._use_skip_connect:
            self._transition_weights_in = \
                    getTransitionMatrixIn(self._shapes, self._size_of_block)
            self._transition_weights_out = \
                    getTransitionMatrixOut(self._shapes, self._size_of_block)
    
            
    def ForwardPass(self):
       assert self._input is not None and self._target is not None

       self._zs = []
       self._as = []
       self._x_normed = {}
       self._bn_mean = {}
       self._bn_var = {}
       self._h = {}
       self._delta_transform_out = None
       
       f_iter = returnZip(self._layers_activations,\
                 self._layers, self._biases, enumerate(self._batchnorm_layers),\
                 self._dropout_layers)
       self._output = self._input
       
       if self._use_skip_connect:
           self._input_to_block = \
                           np.dot(self._input, self._transition_weights_in)
       for elem in f_iter:
           self.DoOneIterationFP(elem)
                       

    def BackwardPass(self):
        self._deltas = []
        self._deltas_bias = []
        self._current_base = None
        self._current_old_base = None
        self._delta_transform_in = None
        
        
        self.loss = np.mean(np.square(self._target - self._output))
        
        self.CalculateBaseDeltas()
        
        

        residual_layers = self._layers[::-1]
        residual_as = self._as[-3::-1] + [self._input]
        
        b_iter = returnZip(residual_as,\
              self._zs[-2::-1], self._layers_activations[-2::-1], \
              residual_layers[:-1], enumerate(self._batchnorm_layers[-2::-1]))
        
        if self._use_skip_connect:
            self._current_old_base = self._current_base
            
            
        for elem in b_iter:
            self.DoOneIterationBP(elem)
                
        if self._use_skip_connect:
            if self._delta_transform_in is None:
                self._delta_transform_in = np.dot(self._input.T, \
                                                  self._current_old_base)

        if self._use_l2_regularization:
            self.DoL2Reg()
                
        if self._use_l1_regularization:
            self.DoL1Reg()
                
            
        if self._clip_gradients_by_value:
            self.DoGradientClipping()
        
        self.DoUpdates()            

        
