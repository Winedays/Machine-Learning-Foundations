# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 23:04:24 2018

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:02:14 2018

@author: USER
"""

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

def test_data( test_x ,test_y , w ) :
    error = 0
    for i in range( len(test_x) ) :
        result = np.dot(test_x[i],w)
        sign = np.sign( result )
        if sign == 0 :
            sign = -1 
        if sign != test_y[i] :
            error += 1
    return error ;

def read_file( file ) :
    file = open('hw1_18_train.dat')
    df = file.readlines()
    #print( df )
    
    #read train data
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
    return x , y 

x , y = read_file('hw1_18_train.dat')

test_x , test_y = read_file('hw1_18_test.dat')

# random PLA with 2000 times
z = list( zip(x , y) )
#update_count_list = []
error_list = [] 
for r in range( 2000 ) :
    print( "Times : %d " % (r) )
    random.Random(r).shuffle( z )
    ran_x , ran_y = zip(*z)    
    
    #print(x)
    ran_x = np.array(ran_x)
    ran_y = np.array(ran_y)
    w = np.zeros( len(ran_x[0]) )
    best_w = [] 
    update_count = 0
    mistakes_count = 0
    min_mistakes_count = len(x) + 2
    # train PLA by Pocke Algorithm
    isTrue = False
    while isTrue != True :
        for i in range( len(x) ) :
            result = np.dot(ran_x[i],w)
            sign = np.sign( result )
            if sign == 0 :
                sign = -1 
            if sign != ran_y[i] :
                w = w + np.multiply( ran_y[i] , ran_x[i] )
                # count the mistakes of new_w
                mistakes_count = test_data( ran_x , ran_y , w )
                update_count += 1
                if min_mistakes_count >= mistakes_count :
                    min_mistakes_count = mistakes_count
                    best_w = w.copy()
            if update_count == 100 :
                print(min_mistakes_count)
                isTrue = True
                break ;
    
    #print(w)
    #print(best_w)
    # test Data
    error = test_data( test_x , test_y , best_w )
    error_list.append( error / len(test_x) )
        
    #update_count_list.append( update_count )
    #if r % 100 == 0 :
        #print( error / len(test_x) )
    
#print( update_count_list )
print( np.mean( error_list ) )

