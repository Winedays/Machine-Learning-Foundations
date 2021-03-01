#encoding=utf8
import sys
import numpy as np
import math
from random import *

# generate input data with 20% flipping noise
def generate_input_data(time_seed):
    np.random.seed(time_seed)
    raw_X = np.sort(np.random.uniform(-1,1,20))
    noised_y = np.sign(raw_X)*np.where(np.random.random(raw_X.shape[0])<0.2,-1,1)
    return raw_X, noised_y

def uniformDistribution() :
    x = np.random.uniform(-1,1,20)
    x = np.sort( x )
    y = np.sign( x )
    for i in range( len(y) ) :
        p = random()
        if p >= 0.8 :
            y[i] = y[i] * -1
    return x, y

def calculate_Ein(x,y):
    # calculate median of interval & negative infinite & positive infinite
    thetas = np.array( [float("-inf")]+[ (x[i]+x[i+1])/2 for i in range(0, x.shape[0]-1) ]+[float("inf")] )
    Ein = x.shape[0]
    sign = 1
    target_theta = 0.0
    # positive and negative rays
    for theta in thetas:
        y_positive = np.where(x>theta,1,-1)
        y_negative = np.where(x<theta,1,-1)
        error_positive = sum(y_positive!=y)
        error_negative = sum(y_negative!=y)
        
        if error_positive>error_negative:
            if Ein>error_negative:
                Ein = error_negative
                sign = -1
                target_theta = theta
        else:
            if Ein>error_positive:
                Ein = error_positive
                sign = 1
                target_theta = theta
                
    # two corner cases
    if target_theta==float("inf"):
        target_theta = 1.0
    if target_theta==float("-inf"):
        target_theta = -1.0
    return Ein, target_theta, sign

def decisionStumpAlgorithm(x,y) :
    theta_set = [-1] 
    sort_x = np.sort( x ) 
    for i in range( len(sort_x)-1 ) :
        theta_set.append( (sort_x[i] + sort_x[i+1]) / 2 )
    theta_set.append(1)
    theta_set = np.array( theta_set )
    best_Ein = len(x) + 2
    best_sign = 0
    best_theta = 0
    for theta in theta_set :
        h_p1 = np.sign( x - theta )
        h_n1 = -1 * np.sign( x - theta )
        err_p1 = np.sum( [ 1 for d in y*h_p1 if d < 0 ] )
        err_n1 = np.sum( [ 1 for d in y*h_n1 if d < 0 ] )

        if err_p1 < err_n1 and err_p1 < best_Ein :
            best_Ein = err_p1 
            best_sign = 1
            best_theta = theta
        elif err_n1 < err_p1 and err_n1 < best_Ein :
            best_Ein = err_p1 
            best_sign = -1
            best_theta = theta
    
    return best_Ein , best_sign , best_theta

def Eout(s,theta) :
    e_out = 0.3*s*( abs(theta)-1 ) + 0.5
    return e_out
 
#if __name__ == '__main__':
#    T = 1000
#    total_Ein = 0.0
#    sum_Eout = 0.0
#    for i in range(0,T):
#        #x,y = generate_input_data(i)
#        x,y = uniformDistribution()
#        #curr_Ein, theta, sign = calculate_Ein(x,y)
#        curr_Ein, sign, theta = decisionStumpAlgorithm(x,y)
#        total_Ein = total_Ein + curr_Ein
#        sum_Eout = sum_Eout + Eout(sign,theta)
#    print( (total_Ein*1.0) / (T*20) )
#    print( (sum_Eout*1.0) / T )
    
if __name__ == "__main__":
    itera = 1000
    e_in = [] 
    e_out = []
    for i in range( itera ) :
        x, y = uniformDistribution()
        best_Ein , best_sign , best_theta = decisionStumpAlgorithm(x,y)
        best_Eout = Eout( best_sign , best_theta )
        e_in.append( best_Ein )
        e_out.append( best_Eout )
    e_in = np.array( e_in ) 
    e_out = np.array( e_out )
    print( 'arr e_in : ' , np.average( e_in ) )
    print( 'arr e_out : ' , np.average( e_out ) )