# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:47:01 2019

@author: ni.martynov
"""

import numpy as np
from abc import ABC, abstractmethod
from initialization import getInit
from activations import getActivation, getDerivation
from utils import (BackwardPassWithBatchNorm, clipByValue, returnListInit, 
                   returnListShape, getOnesForBackProp)

class Base(ABC):
    @abstractmethod
    def __init__(self):
        self._input_shape = None
        self._num_of_neurons = None
        self._layers_init = None
        self._bias_init = None
        self._layers = None
        self._biases = None
        self._batchnorm_layers = None
        self._bn_gammas = None
        self._bn_betas = None
        self._use_bias = None
        self._zs = None
        self._as = None
    
    def InitRegLayers(self):
        self._shapes = [(dim1, dim2) for dim1, dim2 in zip([self._input_shape[1]] + \
                self._num_of_neurons[:-1], self._num_of_neurons)]
        self._shapes.append((self._num_of_neurons[-1],))
        
        
        self._shapes, self._bias_init, self._layers_init, \
                self._layers_activations, self._batchnorm_layers = \
            returnListShape(self._shapes, self._num_of_blocks), \
            returnListInit(self._bias_init, self._num_of_blocks),\
            returnListInit(self._layers_init, self._num_of_blocks),\
            returnListInit(self._layers_activations, self._num_of_blocks),\
            returnListInit(self._batchnorm_layers, self._num_of_blocks)
        
        self._dropout_layers = self._dropout_layers * self._num_of_blocks + \
                                [False]
        
        for init_layer, init_bias, shape in zip(self._layers_init, \
                                                self._bias_init, self._shapes):
            self._layers.append(getInit(init_layer, shape))
            if len(shape) == 2:
                self._biases.append(getInit(init_bias, (shape[1], )))
        self._biases.append(getInit(self._bias_init[-1], ()))
        
        
    def InitBnLayers(self):
        for batchnorm_layer, index_shape in zip(self._batchnorm_layers, \
                            enumerate(self._shapes)):
            index, shape = index_shape
            if batchnorm_layer:
                self._bn_gammas[index] = (getInit(self._batchnorm_gamma_init, \
                                               (shape[1], )))
                self._bn_betas[index] = (getInit(self._batchnorm_beta_init, \
                                              (shape[1], )))  
                    
                
    def DoOneIterationFP(self, elem):
        key, layer, bias, index_bn_layer, drop_layer = elem
        index, bn_layer = index_bn_layer
        if not self._use_bias:
               bias = np.zeros(shape = bias.shape)
        if index == len(self._layers) - 1 and self._use_skip_connect:
            z = np.dot(self._output, layer)
            self._out_transform_comp = self._input_to_block
            if self._num_of_blocks != 1:
                self._input_to_block = np.dot(self._input_to_block,\
                                                  self._transition_weights_out)
            self._zs.append(z + bias \
                            + self._input_to_block)
            a = getActivation(key)(self._zs[-1])
            self._as.append(a)
            if bn_layer:
                self.BnForward(index, drop_layer)
            else:
                self._output = self._as[-1]
                if drop_layer:
                    dropout_mask = np.random.binomial(1, self._keep_prob, size =\
                                        self._output.shape) / self._keep_prob
                    self._output *= dropout_mask
                    self._as[-1] = self._output
            self._input_to_block = self._output
        elif index and (index+1) % self._size_of_block == 0 and \
                len(self._shapes) - index > self._size_of_block and \
                                                self._use_skip_connect:
            z = np.dot(self._output, layer)
            self._zs.append(z + bias \
                            + self._input_to_block)
            a = getActivation(key)(self._zs[-1])
            self._as.append(a)
            if bn_layer:
                self.BnForward(index, drop_layer)
            else:
                self._output = self._as[-1]
                if drop_layer:
                    dropout_mask = np.random.binomial(1, self._keep_prob, size =\
                                        self._output.shape) / self._keep_prob
                    self._output *= dropout_mask
                    self._as[-1] = self._output
            self._input_to_block = self._output
        else:
            self._zs.append(np.dot(self._output, layer) + bias)
            a = getActivation(key)(self._zs[-1])
            self._as.append(a)
            if bn_layer:
                self.BnForward(index, drop_layer)
            else:
                self._output = self._as[-1]
                if drop_layer:
                    dropout_mask = np.random.binomial(1, self._keep_prob, size =\
                                        self._output.shape) / self._keep_prob
                    self._output *= dropout_mask
                    self._as[-1] = self._output
                
    def DoOneIterationBP(self, elem):
        a, z, key, layer, batch_norm_layer_index = elem
        index, layer_bn = batch_norm_layer_index
            
        
        if self._current_base.ndim == 1:
            self._current_base = self._current_base[:, np.newaxis]
        if layer.ndim == 1:
            layer = layer[:, np.newaxis]
        step2 = np.dot(self._current_base, layer.T)
        
        
        if layer_bn:
            step2 = self.MakeUpdateBP(index, step2)
            
        if index and index % self._size_of_block == 0 and \
                                                self._use_skip_connect:
            if self._current_old_base.ndim != 2:
                step2 = step2 + self._current_old_base[:, np.newaxis]
            else:
                step2 = step2 + self._current_old_base
                
            dZ = step2 * getDerivation(key)(z)
            self._current_old_base = step2 * getDerivation(key)(z)
            self._delta_transform_in = np.dot(self._input.T, \
                                dZ) / self._batch_size
        
        self._current_base = step2 * getDerivation(key)(z)
        current_delta = np.dot(a.T, self._current_base) / self._batch_size
        self._deltas.append(current_delta)
            
        if self._use_bias:
            ones_for_bias = np.ones(shape = (self._current_base.shape[0],))
            current_delta_bias = \
                np.dot(self._current_base.T, ones_for_bias) / self._batch_size
            self._deltas_bias.append(current_delta_bias)
        
        if self._use_skip_connect:
            self._delta_transform_out = np.dot(self._out_transform_comp.T, \
                                self._base_for_transform_out) / self._batch_size
             
        
    def MakeUpdateBP(self, index, step2):
        coord = len(self._batchnorm_layers) - 2 - index
        delta_gamma, delta_beta, step2 = \
            BackwardPassWithBatchNorm(step2, self._x_normed[coord], \
            self._batch_size, self._bn_gammas[coord], self._bn_mean[coord], \
            self._bn_var[coord], self._h[coord])
        self._bn_gammas[coord] += self._learning_rate_bn * delta_gamma
        self._bn_betas[coord] += self._learning_rate_bn * delta_beta
        return step2
               
    def BnForward(self, index, dropout):
        self._output = self._as[-1]
        self._h[index] = self._output
        mean = np.mean(self._output, axis=0)
        var = np.var(self._output, axis=0)
        self._output = (self._output - mean) / np.sqrt((var + 1e-08))
        self._x_normed[index] = self._output
        self._bn_mean[index] = mean
        self._bn_var[index] = var
        self._output = self._bn_gammas[index] * self._output + \
                                                        self._bn_betas[index]
        self._as[-1] = self._output
        
        if dropout:
            dropout_mask = np.random.binomial(1, self._keep_prob, size = \
                                         self._output.shape) / self._keep_prob
            self._output *= dropout_mask
        
    def CalculateBaseDeltas(self):
        self._current_base = (self._target - self._output) * \
            getDerivation(self._layers_activations[-1])(self._zs[-1])
        base_delta = \
                np.dot(self._as[-2].T, self._current_base) / self._batch_size
        self._deltas.append(base_delta)
        if self._use_bias:
            base_delta_bias = np.dot(self._current_base, \
                                     np.ones(shape = self._current_base.shape))
            base_delta_bias *= 1. / self._batch_size
            self._deltas_bias.append(base_delta_bias)
        self._base_for_transform_out = self._current_base   
            
        
    def DoL2Reg(self):
        for index in range(len(self._layers)):
            self._layers[index] -= (self._l2_alpha * self._layers[index] * \
                        1. / self._batch_size)
        if self._use_skip_connect:
            self._transition_weights_in -= (self._l2_alpha * \
                        self._transition_weights_in * 1. / self._batch_size)
            if self._num_of_blocks != 1:
                self._transition_weights_out -= (self._l2_alpha * \
                            self._transition_weights_out * 1. / self._batch_size)
        
        
    def DoL1Reg(self):
        for index in range(len(self._layers)):
            self._layers[index] -= (self._l1_beta * \
                    np.sign(self._layers[index]) * 1. / self._batch_size)
        if self._use_skip_connect:
            self._transition_weights_in -= (self._l1_beta * \
                np.sign(self._transition_weights_in) * 1. / self._batch_size)
            if self._num_of_blocks != 1:
                self._transition_weights_out -= (self._l1_beta * \
                np.sign(self._transition_weights_out) * 1. / self._batch_size)
            
    def DoGradientClipping(self):
        self._deltas = [clipByValue(elem, self._min_value_to_clip_gradients, \
                                        self._max_value_to_clip_gradients) \
                            for elem in self._deltas]
        if self._use_bias:
            self._deltas_bias = [clipByValue(elem,\
                                self._min_value_to_clip_gradients, \
                                self._max_value_to_clip_gradients) \
                            for elem in self._deltas_bias]
        if self._use_skip_connect:
            self._transition_weights_in = \
                clipByValue(self._transition_weights_in, \
                            self._min_value_to_clip_gradients, \
                            self._max_value_to_clip_gradients)
            if self._num_of_blocks != 1:
                self._transition_weights_out = \
                    clipByValue(self._transition_weights_out, \
                                self._min_value_to_clip_gradients, \
                                self._max_value_to_clip_gradients)
    
    def DoUpdates(self):
        for index in range(len(self._layers)):
            self._layers[index] += self._learning_rate * self._deltas[-index-1]
            if self._use_bias:
                self._biases[index] += self._learning_rate * \
                            self._deltas_bias[-index-1]  
        if self._use_skip_connect:
            self._transition_weights_in += self._learning_rate * \
                                                    self._delta_transform_in
            if self._num_of_blocks != 1:
                self._transition_weights_out += self._learning_rate * \
                                                        self._delta_transform_out