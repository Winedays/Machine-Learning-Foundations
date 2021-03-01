# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:33:32 2018

@author: USER
"""

import numpy as np

file = open('hw1_15_train.dat')
df = file.readlines()
#print( df )

x = []
y = []
for i in df:
    data = i.split('\t')
    x_d = [ 1. ] 
    x_d += data[0].split()
    x_d = [ float(j) for j in x_d ]
    x.append( x_d )
    y.append( float( data[1].split('\n')[0] ) )
file.close()
    
#print(x)
x = np.array(x)
y = np.array(y)
w = np.zeros( len(x[0]) )

# PLA
counter = 0
isTrue = False
while isTrue != True :
    isTrue = True 
    for i in range( len(x) ) :
        result = np.dot(x[i],w)
        sign = np.sign( result )
        if sign == 0 :
            sign = -1 
        if sign != y[i] :
            #print(y[i])
            #print(x[i])
            w = w + np.multiply( y[i] , x[i] )
            isTrue = False 
            counter += 1
    #print(counter)
            
    if isTrue == True :
        break ;
    
print( counter )
