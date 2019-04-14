##this is a model for question 2,3,4,5/(SGD and  Relu)

import numpy as np
import matplotlib.pyplot as plt


class mod1():
    def __init__(self, N, D, H1, H2):
        self.N = N
        self.D = D
        self.H1 = H1
        self.H2 = H2

        self.reg_mode = None

        self.param = {}
        self.grad = {}
        self._get_param()

        return

    def _get_param(self):
        #define your parameters
        return

    def _check(self, X):
        _, D = X.shape
        if D != self.D:
            raise ValueError(
                "Please check X's dimension is same as W1's shape[0]")
        return

    def _forward(self, X):
        #operation when forward need 
        return

    def _loss(self, X, Y):
        #calculate loss
        return

    def _backward(self, X, Y):
        #backward calculate gradient matrix

        return

    def _optimize(self,lr):
        #SGD method
        return
        

    def _info(self):
        #print some model information
        return

    def _count_params(self):
        #count parameter number
        return

    def train(self, X, Y, lr=0.0001, reg_mode=None, reg=0.1):
        self._forward(X)
        self._loss(X,Y,reg_mode,reg)
        self._backward(X,Y)
        self._optimize(lr)
        return

    def test(self, X):
        return
    
    def cross_entropy(output,Y):
       loss= Y[Y==1].T @ np.log(output[Y==1]) + (1-Y[Y==0].T) @ np.log(1-output[Y==0])
       return loss
