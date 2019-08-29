# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:21:22 2019

@author: ni.martynov
"""

import numpy as np

def getInit(key, shape):
    if key == 'normal':
        return np.random.normal(size=shape)
    
    if key == 'he':
        if len(shape):
            return np.random.normal(size=shape) * np.sqrt(2.0 / shape[0])
        else:
            return np.random.normal(size=shape) * np.sqrt(2.0)
    
    if key == 'xavier':
        if len(shape):
            return np.random.normal(size=shape) * np.sqrt(1.0 / shape[0])
        else:
            return np.random.normal(size=shape)
    
    if key == 'sur':
        if len(shape):
            return np.random.normal(size=shape) * np.sqrt(2.0 / sum([elem for \
                                   elem in shape]))
        else:
            return np.random.normal(size=shape)
    
    if key == 'zero':
        return np.zeros(shape = shape)
    
    if key == 'abs':
        return np.random.normal(size=shape)
    
    if key == 'ones':
        return np.ones(shape = shape)