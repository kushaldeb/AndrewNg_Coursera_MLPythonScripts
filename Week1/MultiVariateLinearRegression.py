#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:58:15 2018

@author: kushaldeb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading data
data = pd.read_csv('ex1data2.txt', header = None)
data = data.as_matrix()

#Segregating data into input and output
X = data[:,(0,1)]
y = data[:,(2)]
y = y.reshape(47,1)
m = len(y)

#print(X[range(0,9), :])

print("First 10 examples from the dataset.\n")
print("X = \n", X[range(0,9), :])
print("y = \n", y[range(0,9), :])

#Now Feature Normalization
mu = np.mean(X, axis = 0)
mu = mu.reshape(1,2)
sigma = np.std(X, axis =0)
sigma = sigma.reshape(1,2)
X = (X-mu)/sigma

#Addind a column of 1 in X
X1 = np.ones((47,1))
X = np.append(X1, X, axis = 1)

#Function for cost function
def CostFunc(X, y, theta):
    J = np.dot(X,theta) - y
    J = (1/(2*m)) * np.dot(J.T, J)
    return J

#Function for Gradient Descent
def GradientDescent(X, y, theta, m, learning_rate, epoochs):
    J_history = np.zeros((epoochs,1))
    for i in range(0,epoochs):
        theta = theta - (learning_rate/m) * (np.dot(X.T, (np.dot(X,theta) - y)))
        J = CostFunc(X, y, theta)
        J_history[i] = J
        if(i%250 == 0):
            print("Cost at iteration#{} : {}".format(i, J))    
    return (theta, J_history)

#Now, Performing gradient descent
#let the initial theta be [[0],[0],[0]]
theta = np.zeros((3,1))
learning_rate = 0.01
epoochs = 501
theta, J_history = GradientDescent(X, y, theta, m, learning_rate, epoochs)

print("\nTheta computed from gradient descent :")
print(theta)

#Plot of the cost function wrt number of iterations
#Plot of the convergence graph
plt.plot(J_history, 'r-')
plt.xlabel('Number of itertions')
plt.ylabel('Cost J')
plt.title('Plot of the cost function wrt number of iterations')
plt.show()

#Predicted price of a 1650 sq-ft, 3 br house
X_temp = np.array([1650, 3])
X_temp = (X_temp-mu)/sigma
X1 = np.ones((1,1))
X_temp = np.append(X1, X_temp, axis = 1)
price = np.dot(X_temp, theta)
print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) is {}\n".format(price))

#Part 2: Calculating theta by Normal Equation
# theta =(X'*X)^(-1) * (X'*y)

theta2 = np.zeros((3,1))
theta2 = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
print("\nTheta computed from normal equations: ")
print(theta2)

#Predicted price of a 1650 sq-ft, 3 br house
price2 = np.dot(X_temp, theta2)
print("Predicted price of a 1650 sq-ft, 3 br house (using normal equation) is {}\n".format(price2))    
