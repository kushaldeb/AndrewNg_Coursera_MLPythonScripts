#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:49:56 2018

@author: kushaldeb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the data
data = pd.read_csv("ex1data1.txt", header = None)
data = data.as_matrix()

#Sergregating data into input(X) and output(Y)
X = data[:,0]
X = X.reshape(97,1)
y = data[:,1]
y = y.reshape(97,1)
m = len(y)

#Plotting the data
plt.scatter(X, y, c="blue", marker="+")
plt.xlabel("Population of city in 10000s")
plt.ylabel("Profit in $10000")
plt.title("Plot of Population vs Profit")
#plt.show()

#Adding a columns of 1 in X for theta0
X1 = np.ones((97,1))
X = np.append(X1, X, axis=1)
#Another way to do this is-:
#X = np.hstack((X1, X))

def CostFunc(X, y, theta, m):
    J = np.dot(X,theta) - y
    J = (1/(2*m))*np.dot(J.T,J)
    return J

def GradientDescent(X, y, theta, m, learning_rate, epoochs):
    for i in range(0,epoochs):
        theta = theta - ( (learning_rate/m)*(np.dot(X.T, (np.dot(X,theta) - y))))
        if(i%250 == 0):
            J = CostFunc(X, y, theta, m)
            print("Cost at iteration#{} : {}".format(i, J))
    return theta

#Lets take the first set of theta to be 0,0
theta = np.zeros((2,1))

#Cost value with theta = (0,0)
J = CostFunc(X, y, theta, m)
print("With theta[0,0]\nCost computed=", J)

#Lets take the first set of theta to be -1,2
theta2 = np.array([[-1],[2]])

#Cost value with theta = (0,0)
J = CostFunc(X, y, theta2, m)
print("With theta[-1,2]\nCost computed=",J)

#Now we will do Gradient Descent
#for theta=(0,0)
learning_rate = 0.01
epoochs = 1500

theta = GradientDescent(X, y, theta, m, learning_rate, epoochs)
print("\nFinal values of theta is:\n", theta)
J = CostFunc(X, y, theta2, m)
print("Final Cost is:\n", J)

#Here 3.5*10000 and 7*10000 are population sizes.
predict1 = np.dot([1, 3.5],theta)
print('\nFor population = 35,000, we predict a profit of {}\n'.format(predict1*10000));
predict2 = np.dot([1, 7],theta)
print('For population = 70,000, we predict a profit of {}\n'.format(predict2*10000));

#Now plotting the linear fit for this graph
y_pred = np.dot(X,theta)
plt.plot(X[:,1], y_pred, 'r-')
plt.show()
