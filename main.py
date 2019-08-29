# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:08:57 2019

@author: ni.martynov
"""
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from perceptron import *
from prepare_data import *
from tqdm import tqdm

errors = []
errors_l2 = []
errors_w = []
a = []
b = []
c = []
d= []

deltas = []
delta = []

if __name__ == '__main__':
    estimator = Perceptron(config)
    estimator.initLayers()
      
    for _ in tqdm(range(config['num_of_epochs'])):
        #print('{e} just started'.format(e = _))
        gen = genBatch(X_train, y_train, config['batch_size'])
        for index, batch in enumerate(gen):
            x, y = batch
            estimator._input = x
            estimator._target = y
            estimator.ForwardPass()
            estimator.BackwardPass()
            a.append(np.linalg.norm(estimator._deltas[0]))
            b.append(np.linalg.norm(estimator._deltas[1]))
            c.append(np.linalg.norm(estimator._deltas[2]))
# =============================================================================
#             if index < 2:
#                 a.append(np.mean(estimator._as[-1]))
#                 b.append(np.mean(estimator._as[-2]))
#                 c.append(np.mean(estimator._as[-3]))
#                 d.append(np.mean(estimator._as[-4]))
#                 print(estimator._layers[-1])
# =============================================================================
            #a.append(np.linalg.norm(estimator._transition_weights_in))
            #b.append(np.linalg.norm(estimator._transition_weights_out))
            #c.append(np.linalg.norm(estimator._layers[2]))
            #d.append(np.linalg.norm(estimator._layers[3]))
            
# =============================================================================
#             deltas.append(np.mean(estimator._deltas[0]))
#             delta.append(np.mean(estimator._deltas[1]))
# =============================================================================
# =============================================================================
#             print(estimator._layers[1])
#             print(estimator._layers[0])
#             print(estimator._layers[2])
#             print(estimator._layers[3])
# =============================================================================
# =============================================================================
#             if index ==10:
#                 break
# =============================================================================
            if index%100==0:
# =============================================================================
#                 print('loss {l} at step {s}'.format(l = estimator.loss,\
#                       s = index))
# =============================================================================
                errors.append(estimator.loss)
# =============================================================================
#                     errors_w.append(abs(estimator.loss - estimator.loss_with_l2))
#                     a.append(np.linalg.norm(estimator._layers[0]))
#                     b.append(np.linalg.norm(estimator._layers[1]))
# =============================================================================
        #print(estimator._bn_gammas)
        #d.append(np.mean(estimator._bn_gammas[1]))
        #a.append(np.linalg.norm(estimator._delta_transform_in))
        #b.append(np.linalg.norm(estimator._delta_transform_out))
        #print(np.linalg.norm(estimator._delta_transform_in))
        #print(np.linalg.norm(estimator._delta_transform_out))
# =============================================================================
#         print('sums')
#         print(estimator._zs)
#         print('weights')
#         print(estimator._layers)
# =============================================================================
        #break
    #print(estimator._layers)     
    plt.title('norms of the weights')
    plt.plot(range(len(a)), a, label='1 layer')
    plt.plot(range(len(b)), b, label='2 layer')
    plt.plot(range(len(c)), c, label='out')
    #plt.plot(range(len(d)), d, label='bias_last')
    plt.legend()
    plt.show()
        
    plt.plot(range(len(errors)), errors, label='loss')
# =============================================================================
#     if estimator._use_l2_regularization:
#         plt.plot(range(len(errors_l2)), errors_l2, label='with l2')
#         plt.plot(range(len(errors_l2)), errors_w, label='with l2 weights')
# =============================================================================
    plt.legend()
    plt.show()
    print(errors[-1])
    
# =============================================================================
#     plt.plot(range(len(deltas)), deltas, label = '1 layer')
#     plt.plot(range(len(delta)), delta, label = '2 layer')
#     plt.legend()
#     plt.show()
# =============================================================================
    
    plt.plot(range(len(d)), d)
    plt.title('norm of gammas')
    plt.show()
    
    print('test is just started')
    gentest = genBatch(X_test, y_test, config['batch_size'])
    estimator._inference = True
    mse = []
    ys = []
    for batch in gentest:
        x, y = batch
        estimator._input = x
        estimator.ForwardPass()
        #res = estimator._output
        res = estimator._output * scale
        #ys.extend(res)
        score = mean_absolute_error(y * scale, res)
        #score = mean_squared_error(y, res)
        mse.append(score)
    print('final score is {s}'.format(s = np.mean(mse)))
    
# =============================================================================
#     a = [(k, v) for k, v in zip(X_test, ys)]
#     a = sorted(a, key = lambda x: x[0])
#     xs, ys = zip(*a)
#     
# # =============================================================================
# #     plt.plot(range(len(errors)), errors, label='Train') 
# #     plt.plot(range(len(mse)), mse, label='test')
# #     plt.legend()
# #     plt.show()
# # =============================================================================
#     
#     
#     plt.scatter(X_train, y_train)
#     plt.plot(xs, ys, c='r')
#     
#     plt.plot(xs, np.power(xs, 3), c = 'g')
#     plt.show()
# =============================================================================

