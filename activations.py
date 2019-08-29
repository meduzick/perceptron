# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:45:50 2019

@author: ni.martynov
"""
import numpy as np

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def Tanh(x):
    return (np.exp(2*x) - 1.0) / (np.exp(2*x) + 1.0)
    
def getActivation(key):
    if key == 'sigmoid':
        return Sigmoid
    if key == 'tanh':
        return Tanh
    if key == 'relu':
        def Relu(x):
            return np.maximum(np.zeros(shape = x.shape), x)
        return Relu
    if key == 'lrelu':
        def Lrelu(x):
            def f(y, alpha = 0.01):
                return  y if y >=0 else y*alpha
            f = np.vectorize(f)
            return f(x)
        return Lrelu
    if key == 'linear':
        def Lin(x):
            return x
        return Lin
    if key == 'elu':
        def Elu(x):
            def f(y, alpha = 1.):
                return y if y > 0 else alpha * (np.exp(y) - 1)
            f = np.vectorize(f)
            return f(x)
        return Elu

def getDerivation(key):
    if key == 'sigmoid':
        def SigmoidDerivation(x):
            return Sigmoid(x)*(1.0 - Sigmoid(x))
        return SigmoidDerivation
    
    if key == 'tanh':
        def TanhDerivation(x):
            return (1.0  - Tanh(x)**2)
        return TanhDerivation
    
    if key == 'relu':
        def ReluDerivative(x):
            def f(y):
                return 0 if y <=0 else 1
            f = np.vectorize(f)
            return f(x)
        return ReluDerivative
    if key == 'lrelu':
        def LreluDerivative(x, alpha = 0.01):
            def f(y):
                return 1 if y > 0 else alpha
            f = np.vectorize(f)
            return f(x)
        return LreluDerivative
    
    if key == 'linear':
        def LinDer(x):
            return np.ones(x.shape)
        return LinDer
    
    if key == 'elu':
        def EluDerivative(x, alpha = 1.):
            def f(y):
                return 1 if y > 0 else alpha * np.exp(y)
            f = np.vectorize(f)
            return f(x)
        return EluDerivative
    