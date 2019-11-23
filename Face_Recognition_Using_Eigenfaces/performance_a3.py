#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:42:17 2019

@author: can
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from plot_a3 import eigen_face_projection

def obtain_new_features(tr_input_col,test_input_col,U,average_face):
    no_tr=tr_input_col.shape[1]
    no_test=test_input_col.shape[1]    
    #train_data_new_features=np.zeros(200*200)
    train_data_new_features=np.zeros(no_tr*no_tr)
    #train_data_new_features=train_data_new_features.reshape(200,200)
    train_data_new_features=train_data_new_features.reshape(no_tr,no_tr)
    for cntr in range(no_tr):
        selected_image=tr_input_col[:,cntr]
        train_data_new_features[cntr,:]=eigen_face_projection(U,selected_image,average_face)
        
    #test_data_new_features=np.zeros(200*200)
    test_data_new_features=np.zeros(no_test*no_tr)
    #test_data_new_features=test_data_new_features.reshape(200,200)
    test_data_new_features=test_data_new_features.reshape(no_test,no_tr)
    for cntr in range(no_test):
        selected_image=test_input_col[:,cntr]
        test_data_new_features[cntr,:]=eigen_face_projection(U,selected_image,average_face)
        
    whole_new_features=np.concatenate((train_data_new_features,test_data_new_features),axis=0)
    
    #input_data=pd.DataFrame(data=whole_new_features)
    return whole_new_features

def calculate_accuracy(whole_new_features,no_of_features):
    no_tr=whole_new_features.shape[1]
    #first no_of_features many features selected.
    partial_features=whole_new_features[:,:no_of_features]
    
    #data_distance is 400*400 distance table. Since First 200 row is training
    #and last 200 is test. we will compare last 200 row with first 200 column.
    data_distance=pd.DataFrame(distance_matrix(partial_features, partial_features))
    #data_distance=data_distance.values[200:,:200]
    data_distance=data_distance.values[no_tr:,:no_tr]
    #after comparing for each row, we will look at the column in which minimum
    #distance is found.In other words, this test sample(row) is similar to the
    #training(column). Since each person has 5 pictures, calculating mod5 will 
    #give us the prediction for this test sample. Note that pictures in training
    #and test data are similar tr0,1,2,3,4 = test0,1,2,3,4
    #                          tr5,6,7,8,9 = test,5,6,7,8,9...
    predictions=np.argmin(data_distance, axis=1)
    which_person=(predictions/5)+1
    
    real_person=(np.arange(40))+1
    real_person=np.tile(real_person,5)
    real_person=np.sort(real_person)
    
    accuracy=sum(which_person==real_person)/float(200)
    return accuracy    
    

