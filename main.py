import numpy as np
import pandas as pd
import os

data = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/train.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init():
    global params,cache
    params = {"W1":np.random.randn(64, 784) * np.sqrt(2. / 784),
              "B1":np.zeros((64, 1)),
              "W2":np.random.randn(10,64) * np.sqrt(2. / 64),
              "B2":np.zeros((10, 1))}
    cache = {}

def sigmoid(n,deriv=False):
    x = 1 / (1 + np.exp(-n))
    if deriv:
        return x * (1 - x)
    return x

def softmax(n):
    return np.exp(n) / np.sum(np.exp(n),axis=0)

def feedforward(A0):
    global cache,params
    cache["A0"] = A0
    cache["Z1"] = params["W1"].dot(A0) + params["B1"]
    cache["A1"] = sigmoid(cache["Z1"])
    cache["Z2"] = params["W2"].dot(cache["A1"]) + params["B2"]
    cache["A2"] = softmax(cache["Z2"])

def getpred(A2):
    return np.argmax(A2,0)

def backprop(DESIRED):
    global cache,params
    m = DESIRED.size
    DESIRED = np.eye(10)[DESIRED].T
    dZ2 = cache["A2"] - DESIRED
    dW2 = 1 / m * dZ2.dot(cache["A1"].T)
    dB2 = 1 / m * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = params["W2"].T.dot(dZ2) * sigmoid(cache["Z1"],True)
    dW1 = 1 / m * dZ1.dot(cache["A0"].T)
    dB1 = 1 / m * np.sum(dZ1,axis=1,keepdims=True)
    return dW1,dB1,dW2,dB2

def updateparams(dW1,dB1,dW2,dB2,lr):
    global params
    params["W1"] -= lr * dW1
    params["B1"] -= lr * dB1
    params["W2"] -= lr * dW2
    params["B2"] -= lr * dB2

def gradientdescent(A0,DESIRED,lr,ITERS):
    init()
    for i in range(ITERS):
        feedforward(A0)
        dW1,dB1,dW2,dB2 = backprop(DESIRED)
        updateparams(dW1,dB1,dW2,dB2,lr)
        if i % 50 == 0:
            print(i,"iters done.")

gradientdescent(X_train,Y_train,0.1,500)
feedforward(X_train[:,0,None])

correct = 0
for i in range(100):
    feedforward(X_dev[:,i,None])
    correct += np.argmax(cache["A2"],0) == Y_dev[i]
    print(f"Networks prediction : {np.argmax(cache["A2"],0)}")
    print(f"Real answer : {Y_dev[i]}")
    print(f"Current accuracy is %{correct/i*100}")
