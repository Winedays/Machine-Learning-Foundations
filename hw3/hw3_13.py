# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:08:06 2019

@author: USER
"""
import numpy as np
import random
import matplotlib.pyplot as plt

def uniformDistribution() :
    N = 1000
    x1 = np.random.uniform(-1,1,N)
    x2 = np.random.uniform(-1,1,N)
    #x = np.sort( x )
    y = np.sign( x1**2 + x2**2 - 0.6 )
    # 10% nosie & set x
    x = []
    for i in range( len(y) ) :
        x.append( [x1[i] , x2[i]] )
        p = random.random()
        if p >= 0.9 :
            y[i] = y[i] * -1
    x = np.array(x)
    y = np.array(y)
    return x, y

def linearRegression(x,y) :
    weight = np.dot( np.dot( np.linalg.inv(np.dot(x.T,x)),x.T ) , y )
    return weight
    
def Ein(weight, x, y) :
    e_in = x.dot(weight) * y
    e_in = np.sum( np.sign(e_in) < 0 ) / len(y)
    return np.average(e_in)

#Q14
def transform_ein(weight, x, y) :
    x_t = x.T
    x1 = x_t[0]
    x2 = x_t[1]
    transform = -1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*(x1**2) + 1.5*(x2**2)
    e_in = np.sum( np.sign(transform*y) < 0 ) / len(y)
    return np.average(e_in)

if __name__ == "__main__":
    iterate = 1000
    E_in = [] 
    E_out = []
    
    for i in range(iterate) :
        x,y = uniformDistribution()
        weight = linearRegression(x,y)
        #e_in = Ein(weight, x, y)
        e_in = transform_ein(weight, x, y)
        E_in.append( e_in )
        #Q15
        test_x,test_y = uniformDistribution()
        test_weight = linearRegression(test_x,test_y)
        e_out = transform_ein(test_weight, test_x, test_y)
        E_out.append(e_out)
    print( 'average e_in =' , np.average(E_in) )
    print( 'average e_out =' , np.average(E_out) )