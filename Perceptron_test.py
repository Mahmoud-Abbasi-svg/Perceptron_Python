# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 02:54:56 2021

@author: HP_PC2
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from Perceptron_Python import Perceptron

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, 
                           cluster_std=1.05, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)

predictions = p.predict(X_test)

print("Perceptron accuracy is: ", accuracy(y_test, predictions))


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:,0], X_train[:,1], marker='o', c=y_train)



















