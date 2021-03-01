# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:10:26 2018

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:33:32 2018

@author: USER
"""

import numpy as np
import random 

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
    
# random PLA with 2000 times
z = list( zip(x , y) )
counter_list = []
for r in range( 2000 ) :
    #print( "Times : %d " % (r) )
    random.Random(r).shuffle( z )
    ran_x , ran_y = zip(*z)    
    
    #print(x)
    ran_x = np.array(ran_x)
    ran_y = np.array(ran_y)
    w = np.zeros( len(ran_x[0]) )
    
    # PLA
    counter = 0
    isTrue = False
    while isTrue != True :
        isTrue = True 
        for i in range( len(x) ) :
            result = np.dot(ran_x[i],w)
            sign = np.sign( result )
            if sign == 0 :
                sign = -1 
            if sign != ran_y[i] :
                #print(y[i])
                #print(x[i])
                w = w + np.multiply( ran_y[i] , ran_x[i] )
                isTrue = False 
                counter += 1
        if isTrue == True :
            break ;
    counter_list.append( counter )
    #if r % 100 == 0 :
    #    print( w )
    
#print( counter_list )
print( np.mean( counter_list ) )

