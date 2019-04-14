##this is a model for question 1/(Adam and leaky Relu)

import numpy as np
import matplotlib.pyplot as plt


def leaky_relu(array):
    new = np.zeros(array.shape)
    new += array
    new[new<0]*=0.2
    return new

class mod1():
    def __init__(self, N, D, H1, H2, reg_mode=None):
        self.N = N
        self.D = D
        self.H1 = H1
        self.H2 = H2
        self.reg_mode = reg_mode
        

        self.param = {}
        self.grad = {}
        self._get_param()
        self._get_adam_param()

        return

    def _get_param(self, std=1e-1):
        np.random.seed(0)
        self.param["W1"] = std * np.random.randn(self.D, self.H1)
        self.param["W2"] = std * np.random.randn(self.H1, self.H2)
        self.param["W3"] = std * np.random.randn(self.H2, 2)
        self.param["b1"] = np.zeros(self.H1)
        self.param["b2"] = np.zeros(self.H2)
        self.param["b3"] = np.zeros(2)

        return
    
    def _get_adam_param(self):
        self.W1_m = 0
        self.W2_m = 0 
        self.W3_m = 0
        
        self.W1_v = 0
        self.W2_v = 0
        self.W3_v = 0
        
        self.b1_m = 0
        self.b2_m = 0
        self.b3_m = 0
        
        self.b1_v = 0
        self.b2_v = 0
        self.b3_v = 0
        self.t = 0


    def _check(self, X):
        _, D = X.shape
        if D != self.D:
            raise ValueError(
                "Please check X's dimension is same as W1's shape[0]")
        return

    def _forward(self, X):
        b, D = X.shape
        self.layer_1 = X @ self.param["W1"] + self.param["b1"]
        self.layer_1_o = leaky_relu(self.layer_1)
        self.layer_2 = self.layer_1_o @ self.param["W2"] + self.param["b2"]
        self.layer_2_o = leaky_relu(self.layer_2)
        self.layer_3 = self.layer_2_o @ self.param["W3"] + self.param["b3"]
        self.f = self.layer_3 - np.max(self.layer_3)
        self.divider = np.sum(np.exp(self.f), axis=1, keepdims=True)
        # self.divider[self.divider==0] = 1
        self.output = np.exp(self.f) / self.divider

        return self.output

    def _loss(self, Output, Y):
        n,D = Output.shape
        if self.reg_mode is None:
            reg_term = 0
        if self.reg_mode == "L1":
            reg_term = self.reg * (np.sum(self.param["W1"]) + np.sum(
                self.param["W2"]) + np.sum(self.param["W3"]))
        if self.reg_mode == "L2":
            reg_term = 1 / 2 * self.reg * (
                np.sum(self.param["W1"] * self.param["W1"]) + np.sum(
                    self.param["W2"] * self.param["W2"]) + np.sum(
                        self.param["W3"] * self.param["W3"]))

        loss = self.cross_entropy(Output, Y) + reg_term
        self.loss = loss / n
        return self.loss

    def _backward(self, X, Y):
        n,D = X.shape
        dW1 = np.zeros(self.param["W1"].shape)
        dW2 = np.zeros(self.param["W2"].shape)
        dW3 = np.zeros(self.param["W3"].shape)
        db1 = np.zeros(self.param["b1"].shape)
        db2 = np.zeros(self.param["b2"].shape)
        db3 = np.zeros(self.param["b3"].shape)
        d_R = np.zeros(Y.shape)
        d_output = np.zeros(Y.shape)

        # print("output: ",self.output)
        # print("--------------")
        d_output[Y == 1] = self.output[Y == 1] - 1
        d_output[Y == 0] = self.output[Y == 0]

        # print("layer2out=",self.layer_2_o)
        # print("---------------")

        dW3 = self.layer_2_o.T @ (d_output)
        # print("dw3=",dW3)
        # print("---------------")
        db3 = np.sum(d_output, axis=0)

        dh3_o = (d_output) @ self.param["W3"].T

        dh3 = dh3_o
        dh3[self.layer_2 <= 0] *=0.2

        dW2 = self.layer_1_o.T @ dh3
        db2 += np.sum(dh3, axis=0)
        # print("dw2=",dW2)
        # print("---------------")

        dh2_o = dh3 @ self.param['W2'].T
        dh2 = dh2_o
        dh2[self.layer_1 <= 0] *=0.2

        dW1 = X.T @ dh2
        db1 += np.sum(dh2, axis=0)

        reg_term = {}

        if self.reg_mode == "L1":
            reg_term['W3'] = self.reg
            reg_term['W2'] = self.reg
            reg_term["W1"] = self.reg
        if self.reg_mode == "L2":
            reg_term["W3"] = self.reg * self.param["W3"]
            reg_term["W2"] = self.reg * self.param["W2"]
            reg_term["W1"] = self.reg * self.param["W1"]
        if self.reg_mode is None:
            reg_term["W3"] = 0
            reg_term['W2'] = 0
            reg_term["W1"] = 0

        self.grad["W3"] = (dW3 + reg_term['W3']) / n
        self.grad["b3"] = db3 / n
        self.grad["W2"] = (dW2 + reg_term['W2']) / n
        self.grad["b2"] = db2 / n
        self.grad["W1"] = (dW1 + reg_term['W1']) / n
        self.grad["b1"] = db1 / n

        return self.loss, self.grad

    def _optimize(self, lr, mode="SGD", beta1=0.99, beta2=0.996, epilson=1e-9):
        self.lr = lr
        if mode == "SGD":
            self.param["W3"] += -self.grad["W3"] * self.lr
            self.param["b3"] += -self.grad["b3"] * self.lr
            self.param["W2"] += -self.grad["W2"] * self.lr
            self.param["b2"] += -self.grad["b2"] * self.lr
            self.param["W1"] += -self.grad["W1"] * self.lr
            self.param["b1"] += -self.grad["b1"] * self.lr

        if mode == "Adam":
            self.t += 1  # Increment Time Step

            self.W3_m = self.W3_m * beta1 + (1 - beta1) * self.grad["W3"]
            self.W2_m = self.W2_m * beta1 + (1 - beta1) * self.grad["W2"]
            self.W1_m = self.W1_m * beta1 + (1 - beta1) * self.grad["W1"]
            self.b3_m = self.b3_m * beta1 + (1 - beta1) * self.grad["b3"]
            self.b2_m = self.b2_m * beta1 + (1 - beta1) * self.grad["b2"]
            self.b1_m = self.b1_m * beta1 + (1 - beta1) * self.grad["b1"]

            self.W3_v = self.W3_v * beta2 + (1-beta2) * (self.grad["W3"]**2)
            self.W2_v = self.W2_v * beta2 + (1-beta2) * (self.grad["W2"]**2)
            self.W1_v = self.W1_v * beta2 + (1-beta2) * (self.grad["W1"]**2)
            self.b3_v = self.b3_v * beta2 + (1-beta2) * (self.grad["b3"]**2)
            self.b2_v = self.b2_v * beta2 + (1-beta2) * (self.grad["b2"]**2)
            self.b1_v = self.b1_v * beta2 + (1-beta2) * (self.grad["b1"]**2)

            W3_m_corrected = self.W3_m / (1 - (beta1**self.t))
            W3_v_corrected = self.W3_v / (1 - (beta2**self.t))

            W2_m_corrected = self.W2_m / (1 - (beta1**self.t))
            W2_v_corrected = self.W2_v / (1 - (beta1**self.t))

            W1_m_corrected = self.W1_m / (1 - (beta1**self.t))
            W1_v_corrected = self.W1_v / (1 - (beta1**self.t))

            b3_m_corrected = self.b3_m / (1 - (beta1**self.t))
            b3_v_corrected = self.b3_v / (1 - (beta1**self.t))

            b2_m_corrected = self.b2_m / (1 - (beta1**self.t))
            b2_v_corrected = self.b2_v / (1 - (beta1**self.t))

            b1_m_corrected = self.b1_m / (1 - (beta1**self.t))
            b1_v_corrected = self.b1_v / (1 - (beta1**self.t))

            w3_update = W3_m_corrected / (np.sqrt(W3_v_corrected)+epilson)
            w2_update = W2_m_corrected / (np.sqrt(W2_v_corrected)+epilson)
            w1_update = W1_m_corrected / (np.sqrt(W1_v_corrected)+epilson)

            b3_update = b3_m_corrected / (np.sqrt(b3_v_corrected)+epilson)
            b2_update = b2_m_corrected / (np.sqrt(b2_v_corrected)+epilson)
            b1_update = b1_m_corrected / (np.sqrt(b1_v_corrected)+epilson)
            
            self.param["W3"] += -w3_update * self.lr
            self.param["b3"] += -b3_update * self.lr
            self.param["W2"] += -w2_update * self.lr
            self.param["b2"] += -b2_update * self.lr
            self.param["W1"] += -w1_update * self.lr
            self.param["b1"] += -b1_update * self.lr
        # if mode == "RMSprop":
        #     self.param["W3"] += -self.grad["W3"] * self.lr
        #     self.param["b3"] += -self.grad["b3"] * self.lr
        #     self.param["W2"] += -self.grad["W2"] * self.lr
        #     self.param["b2"] += -self.grad["b2"] * self.lr
        #     self.param["W1"] += -self.grad["W1"] * self.lr
        #     self.param["b1"] += -self.grad["b1"] * self.lr
        return

    def loss_f(self, X, Y):
        b, D = X.shape
        self.layer_1 = X @ self.param["W1"] + self.param["b1"]
        self.h1 = np.maximum(0, self.layer_1)  #relu
        self.layer_2 = self.h1 @ self.param["W2"] + self.param["b2"]
        self.h2 = np.maximum(0, self.layer_2)  #relu
        self.layer_3 = self.h2 @ self.param["W3"] + self.param["b3"]
        self.layer_3 -= np.max(self.layer_3)
        self.divider = np.sum(np.exp(self.layer_3), axis=1, keepdims=True)
        # self.divider[self.divider==0] = 1
        self.output = np.exp(self.layer_3) / self.divider

        loss = -1 * np.sum(np.log(self.output[Y == 1]))
        self.loss = loss / self.N

        dW1 = np.zeros(self.param["W1"].shape)
        dW2 = np.zeros(self.param["W2"].shape)
        dW3 = np.zeros(self.param["W3"].shape)
        db1 = np.zeros(self.param["b1"].shape)
        db2 = np.zeros(self.param["b2"].shape)
        db3 = np.zeros(self.param["b3"].shape)

        d_R = np.zeros(Y.shape)
        d_output = np.zeros(Y.shape)

        d_R = self.output
        d_R[Y == 1] -= 1
        d_R = d_R / self.N

        dW3 = self.h2.T @ (d_R)

        db3 = np.sum(d_R, axis=0)
        # print(db3.shape)

        dh2 = (d_R) @ self.param["W3"].T

        dh2[self.h2 <= 0] = 0

        dW2 = self.h1.T @ dh2

        db2 = np.sum(dh2, axis=0)

        dh1 = dh2 @ self.param['W2'].T
        dh1[self.h1 <= 0] = 0

        dW1 = X.T @ dh1
        db1 = np.sum(dh1, axis=0)

        reg_term = {}

        if self.reg_mode == "L1":
            reg_term['W3'] = 1
            reg_term['W2'] = 1
            reg_term["W1"] = 1
        if self.reg_mode == "L2":
            reg_term["W3"] = np.sum(self.param["W3"])
            reg_term["W2"] = np.sum(self.param["W2"])
            reg_term["W1"] = np.sum(self.param["W1"])
        if self.reg_mode is None:
            reg_term["W3"] = 0
            reg_term['W2'] = 0
            reg_term["W1"] = 0

        self.grad["W3"] = dW3
        self.grad["b3"] = db3
        self.grad["W2"] = dW2
        self.grad["b2"] = db2
        self.grad["W1"] = dW1
        self.grad["b1"] = db1

        return self.loss, self.grad

    def _info(self):
        print(self.param)
        return

    def _count_params(self):
        total_number = np.prod(self.param["W3"].shape) + np.prod(
            self.param["W2"]) + np.prod(self.param["W1"]) + np.prod(
                self.param["b1"].shape) + np.prod(
                    self.param["b2"].shape) + np.prod(self.param["b3"].shape)
        print("total variable number = ", total_number)
        return

    def train(self, X, Y, lr=0.0001, reg_mode=None, reg=0.1):
        self.reg = reg
        output = self._forward(X)
        loss = self._loss(output, Y)
        self._backward(X, Y)
        self._optimize(lr,mode="Adam")
        accuracy = self.accuracy_f(output, Y)
        return loss, accuracy, output

    def test(self, X, Y):
        test = self._forward(X)

        loss_t = self._loss(test, Y)

        acc_t = self.accuracy_f(test, Y)

        return loss_t, acc_t, test

    def cross_entropy(self, output, Y):
        # print(output)
        output[(output == 0) & (Y == 1)] += 1e-72
        # print("after:",output)
        loss = -1 * np.sum(np.log(output[Y == 1]))
        # print("loss: ",loss)
        return loss

    def accuracy_f(self, output, Y):
        # print("output:", output)
        index = np.argmax(output, axis=1)
        Y_index = np.argmax(Y, axis=1)
        accuracy = np.sum((index - Y_index) == 0) / index.shape[0]
        return accuracy
