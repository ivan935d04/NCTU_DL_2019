import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

class three_layer_model():
    def __init__(self,input_size,hidden_size_first,hidden_size_second,output_size,weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size_first)
        self.params['b1'] = np.zeros(hidden_size_first)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size_first, hidden_size_second) 
        self.params['b2'] = np.zeros(hidden_size_second)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size_second, output_size) 
        self.params['b3'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = softmax(a2)
        a3 = np.dot(z2, W3) + b2
        y = softmax(a3)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)

    def cross_entropy_error(self,y,t):
        delta = 1e-8
        return -np.sum(t*np.log(y+delta))
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def sigmoid_grad(self, x):
        return np.exp(-x) / ((1+np.exp(-x))^2)

    def softmax(self, a):
        C = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def gradient(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b2
        y = softmax(a3)
        
        # backward
        dL_dy = (y - t) / batch_num

        grads['dL_dW3'] = np.dot(z2.T, dL_dy)
        grads['dL_db3'] = np.sum(dL_dy, axis=0)
        
        dz2 = np.dot(dy, W3.T)
        da2 = sigmoid_grad(a2) * dz2

        grads['dL_dW2'] = np.dot(z1.T, dy)
        grads['dL_db2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['dL_dW1'] = np.dot(x.T, da1)
        grads['dL_db1'] = np.sum(da1, axis=0)

        return grads