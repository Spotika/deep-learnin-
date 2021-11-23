from os import read
import numpy as np
import random

# Usefull functions------------
# 1st - RELU
# 2nd - Softmax
def Relu(x): # 1
    return np.maximum(x, 0)
def Softmax(x): # 2
    e = np.exp(x)
    return e / e.sum()
        


class Network():
    


    # initialization hyper PARAMS + add weights
    def __init__(self, HYPER_PARAMS):
        self.PARAMS = {
            'IN_DIM' : 0,
            'OUT_DIM' : 0,
            'HIDDEN_L' : [],
        }
        # Hyper ADD
        for key in HYPER_PARAMS.keys():
            self.PARAMS[key] = HYPER_PARAMS[key]




        # Weights ADD
        self.WEIGHTS = [] # if weights.txt contain smth else gen random weights
        # from file
        try:
            with open('weights.txt', 'r') as w:
                RECIVED = w.read()
        except FileNotFoundError:
            with open('weights.txt', 'w') as w:
                pass
            RECIVED = ''

        # gen new weights
        if RECIVED == '':
            # for first input
            self.WEIGHTS.append([np.random.randn(self.PARAMS['IN_DIM'], self.PARAMS['HIDDEN_L'][0]), np.random.randn(self.PARAMS['HIDDEN_L'][0])])
            # for middle input
            for i in range(1, len(self.PARAMS['HIDDEN_L']) - 1):
                self.WEIGHTS.append([np.random.randn(self.PARAMS['HIDDEN_L'][i]), np.random.randn(self.PARAMS['HIDDEN_L'][i+1])])
            # for latest input
            self.WEIGHTS.append([ np.random.randn(self.PARAMS['HIDDEN_L'][-1], self.PARAMS['OUT_DIM']), np.random.randn(self.PARAMS['OUT_DIM']) ])
        else: # else replace from file
            pass




    # Prediction function
    def predict(self, x):
        self.layers = [] # list of []+[t_N, h_N]
        self.layers.append([0, x]) # input data

        for i in range(len(self.PARAMS['HIDDEN_L'])+1): # hidden [t_n, h_n]
            self.layers.append([0, 0])

        for i in range(1, len(self.layers)):
            # self.layers[i][0] -> t_i, self.layers[i][1] -> h_i
            # self.WEIGHTS[i][0] -> W_i, self.WEIGHTS[i][1] -> b_i
            self.layers[i][0] = self.layers[i-1][1] @ self.WEIGHTS[i-1][0] + self.WEIGHTS[i-1][1] # t_i = h_(i-1) @ W_(i-1) + b_(i-1)
            if i == len(self.layers)-1: # for latest element (result)
                self.Z = Softmax(self.layers[i][0])
            else:
                self.layers[i][1] = Relu(self.layers[i][0]) # h_i = Relu(t_i)
        
        # Index of maximum element
        self.Prediction = np.argmax(self.Z)
        return self.Prediction


    # return the weights
    def get_weights(self):
        return self.WEIGHTS

    # save weights to weigth.txt 
    # def save_weights(self):
    #     with open('weights.txt', 'w') as w:
    #         w.write(str(self.WEIGHTS))
        















    # def Sparse_cross_entropy(z, y):
    #     return -np.log(z[0, y])
    # def To_full(y, num):
    #     y_full = np.zeros((1, num))
    #     y_full[0, y] = 1
    #     return y_full

    # def relu_deriv(t):
    #     return (t >= 0).astype(float)


        
    # def predict(x):
    #         t1 = x @ W1 + b1
    #         h1 = Relu(t1)
    #         t2 = h1 @ W2 + b2
    #         z = Softmax(t2)
    #         return z

    # include 
    # INPUT_DIM = 4
    # OUTPUT_DIM = 3
    # H_DIM = 10
    # ALPHA = 0.001
    # NUM_EPOCH = 400


    # loss = []
    # for _ in range(NUM_EPOCH):
    #     random.shuffle(dataset)

    #     for i in dataset:
    #         x, y = i
    #         # Foward
    #         t1 = x @ W1 + b1
    #         h1 = Relu(t1)
    #         t2 = h1 @ W2 + b2
    #         z = Softmax(t2)
    #         E = Sparse_cross_entropy(z, y)

    #         # Backward
    #         y_full = To_full(y, OUTPUT_DIM)
    #         dt2 = z - y_full
    #         dW2 = h1.T @ dt2
    #         db2 = dt2
    #         dh1 = dt2 @ W2.T
    #         dt1 = dh1 * relu_deriv(t1)
    #         dW1 = x.T @ dt1
    #         db1 = dt1

    #         # Update
    #         W1 = W1 - ALPHA * dW1
    #         b1 = b1 - ALPHA * db1
    #         W2 = W2 - ALPHA * dW2
    #         b2 = b2 - ALPHA * db2
    #         loss.append(E)
