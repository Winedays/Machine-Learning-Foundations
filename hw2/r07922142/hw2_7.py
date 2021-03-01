# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 01:42:31 2018

@author: USER
"""

import numpy as np
import random
import matplotlib.pyplot as plt

def uniformDistribution() :
    x = np.random.uniform(-1,1,20)
    x = np.sort( x )
    y = np.sign( x )
    for i in range( len(y) ) :
        p = random.random()
        if p >= 0.8 :
            y[i] = y[i] * -1
    return x, y

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
    
    return best_Ein / len(x) , best_sign , best_theta
    
    
def Eout(s,theta) :
    e_out = 0.3*s*( abs(theta)-1 ) + 0.5
    return e_out

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
    
    E_show = e_in-e_out
    #bins = [5,15,25,35,45,55,65,75,85] 
    bins = np.arange( -40 , 40 ) / 100
    #bins = range( min(E_show), max(E_show) + binwidth, binwidth)
    plt.hist( E_show , bins=bins , facecolor='#9999ff' , edgecolor='white')
    plt.title("E_in - E_out histogram") 
    plt.xlabel('E_in - E_out')
    plt.ylabel('Counter')
    
    fig = plt.gcf()
    fig.set_size_inches(12.5, 7.2)
    fig.savefig('histogram.png', dpi=100)
    #plt.savefig('histogram.png')
    plt.show()
