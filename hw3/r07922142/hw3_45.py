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
    learn_rate_4 = 0.01
    learn_rate_5 = 0.001
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
    gd_ein = []
    gd_eout = []
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
        weight = weight - learn_rate_4 * grad_W
        # Error
        e_in = get_error(weight, train_x, train_y)
        e_out = get_error(weight, test_x, test_y)
        gd_ein.append(e_in)
        gd_eout.append(e_out)
        
#    print(gd_ein[-1])
#    print(gd_eout[-1])
        
    #Q20
    weight = np.zeros( len(train_x[0]) )
    sgd_ein = []
    sgd_eout = []
    for i in range( itera ) :
        choice = i % 1000
        #for j in range( len(train_x) ) :
        grad_W = calculate_gradient(weight, train_x[choice], train_y[choice])
            
        #update
        weight = weight - learn_rate_5 * grad_W
        # Error
        e_in = get_error(weight, train_x, train_y)
        e_out = get_error(weight, test_x, test_y)
        sgd_ein.append(e_in)
        sgd_eout.append(e_out)
    
#    print(sgd_ein[-1])
#    print(sgd_eout[-1])
    
    plt.plot( gd_ein , label='gradient descent')
    plt.plot( sgd_ein , label=' stochastic gradient descent')
    plt.xlabel( "t" )
    plt.ylabel( "E_in" )
    plt.title( "E_in of gradient descent version and the stochastic gradient descent" )
    #plt.title( "Top-N" )
    plt.legend(loc='lower left')
    fig = plt.gcf()
    fig.set_size_inches(12.5, 7.2)
    fig.savefig('./e_in.png', dpi=100)
    plt.show()
    plt.close()
        
    plt.plot( gd_eout , label='gradient descent')
    plt.plot( sgd_eout , label=' stochastic gradient descent')
    plt.xlabel( "t" )
    plt.ylabel( "E_out" )
    plt.title( "E_out of gradient descent version and the stochastic gradient descent" )
    #plt.title( "Top-N" )
    plt.legend(loc='lower left')
    fig = plt.gcf()
    fig.set_size_inches(12.5, 7.2)
    fig.savefig('./e_out.png', dpi=100)
    plt.show()
    plt.close()
    
    
    
    
    
