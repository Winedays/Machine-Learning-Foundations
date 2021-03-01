# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 19:06:52 2019

@author: USER
"""
import sys
import numpy as np
import random
import matplotlib.pyplot as plt

def readFile( filepath ) :
    file = open(filepath)
    df = file.readlines()
    #print( df )
    
    x = []
    y = []
    for i in df:
        data = i.split(' ')
        x_d = data[1:-1] # not include data[-1]
        x.append( [ float(d) for d in x_d ] )
        y.append( int( data[-1].split('\n')[0] ) )
    file.close()
    x = np.array(x)
    y = np.array(y)
    return x,y

# calculate graident of Ein
def calculate_gradient(w, x, y):
    s = np.dot(w, x.transpose())*y
    theta = 1.0/(1+np.exp(s))
    gradient_all = theta.reshape(-1,1)*(-1)*y.reshape(-1,1)*x
    gradient_average = np.sum(gradient_all, axis=0)
    return gradient_average / gradient_all.shape[0]

def get_error(w, x, y) :
    z = np.dot(x,w)
    sigma = 1 / (1 + np.exp(-1 * z))
    result = []
    for data in sigma :
        if data >= 0.5 :
            result.append(1)
        else :
            result.append(-1)
    error = np.sum( [ 1 for i in range(len(result)) if result[i] != y[i] ] ) / len(y)
    return error

if __name__ == "__main__":
    itera = 2000
    learn_rate = 0.001
    e_in = [] 
    e_out = []
    
    train_file = './hw3_train.dat'
    test_file = './hw3_test.dat'
    if len(sys.argv) == 3 :
        train_file = sys.argv[1]
        test_file = sys.argv[2]
    
    train_x , train_y = readFile( train_file )
    test_x , test_y = readFile( test_file )
    #print( train_x , train_y )
    # add '1' column
    train_x = np.hstack((np.ones(train_x.shape[0]).reshape(-1,1),train_x))
    test_x = np.hstack((np.ones(test_x.shape[0]).reshape(-1,1),test_x))
    weight = np.zeros( len(train_x[0]) )
    
    error = 0
    for i in range( itera ) :
#        z = np.dot(train_x,weight)
#        sigma = 1 / (1 + np.exp(-1 * z))
#
#        zw = train_y * np.dot(train_x,weight)
#        theta = 1 / (1 + np.exp(-1 * zw))
#        grad_W = np.average( theta.reshape(-1,1) * ( -1 * train_y.reshape(-1,1) * train_x ) )
        #grad_W = np.mean(-1 * train_x * (np.squeeze(train_y) - sigma).reshape((len(train_x),1)), axis=0)
        grad_W = calculate_gradient(weight, train_x, train_y)
        
        #update
        weight = weight - learn_rate * grad_W
        
    error = get_error(weight, train_x, train_y)
    print( error )
    print( weight )
    
    error = get_error(weight, test_x, test_y)
    print( error )
    
    #Q20
    weight = np.zeros( len(train_x[0]) )
    for i in range( itera ) :
        for j in range( len(train_x) ) :
            grad_W = calculate_gradient(weight, train_x[j], train_y[j])
            
            #update
            weight = weight - learn_rate * grad_W
        
    error = get_error(weight, train_x, train_y)
    print( grad_W.shape )
    print( error )
    print( weight )
    
    error = get_error(weight, test_x, test_y)
    print( error )
    
