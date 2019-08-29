# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:20:03 2019

@author: ni.martynov
"""
import numpy as np


def clipByValue(grads, min_value, max_value):
    grads = np.maximum(grads, np.full(shape = grads.shape, fill_value = min_value))
    return np.minimum(grads, np.full(shape = grads.shape, fill_value = max_value))

def BackwardPassWithBatchNorm(previous_step, x_normed, N, gamma, mean, var, h):
    dgamma = np.sum(previous_step * x_normed, axis=0)
    dbeta = np.sum(previous_step, axis=0)
    
    std_inv = 1. / np.sqrt(var + 1e-08)
    out = 1. / N * gamma * std_inv * (N * previous_step - np.sum(previous_step, \
        axis = 0) - std_inv * (h - mean) * np.sum(previous_step * (h - mean), axis = 0))   
    
    return dgamma, dbeta, out

def CalculateRunningVals(running_mean, running_var, current_mean, current_var):
    if running_mean is None:
        return current_mean, current_var
    else:
        return 0.9 * running_mean + 0.1 * current_mean, 0.9 * running_var + \
                        0.1 * current_var
                        
def returnZip(*args):
    return zip(*args)

def returnListShape(l, num_of_blocks):
    block = l[1:-1]
    transition = (block[-1][-1], block[0][0])
    block.append(transition)
    
    return l[:1] + block*(num_of_blocks - 1) + l[1:]

def returnListInit(l, num_of_blocks):
    block = l[1:-1]
    transition = l[-1]
    block.append(transition)
    
    return l[:1] + block*(num_of_blocks - 1) + l[1:]

def getTransitionMatrixIn(shapes, block_size):
    shape = (shapes[0][0], shapes[block_size-1][-1]) if \
                len(shapes[block_size])>1 else (shapes[0][0],) 
    return np.random.normal(size = shape)  

def getTransitionMatrixOut(shapes, block_size):
    shape = (shapes[block_size-1][1], )
    return np.random.normal(size = shape)                                    

def getOnesForBackProp(old_base, step2):
    if len(old_base.shape) == 2:
        shape = (old_base.shape[1], step2.shape[1])
    else:
        shape = (step2.shape[1], )
    return np.ones(shape)
