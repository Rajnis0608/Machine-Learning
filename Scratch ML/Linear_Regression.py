# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 20:41:14 2022

@author: GOD
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initializing parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights = self.weights - self.lr * dw
            self.bias =self.bias - self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

X,y = datasets.make_regression(n_samples=100,n_features=2,noise=20,random_state=4)
X_train,X_test,Y_train,Y_test = train_test_split(X,y,random_state=1)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
predicted = regressor.predict(X_test)

def mse(y_true,y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value = mse(Y_test,predicted)
print(mse_value)