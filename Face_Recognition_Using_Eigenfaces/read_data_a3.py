#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 13:33:32 2019

@author: can
"""


import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def prepare_train_test(path_to_data):
    no_of_training=200
    no_of_test=200
    
    training_input=[None]*no_of_training
    training_class=[None]*no_of_training
    test_input=[None]*no_of_test
    test_class=[None]*no_of_test
    
    instances = glob.glob(path_to_data+"//*")
    tr_counter=0
    test_counter=0
    
    for inst in instances:
        if inst[-3:]!="TXT":
            pers_no=inst[-3:]
            if pers_no[0]=="s":
                pers_no=int(pers_no[-2:])
            else:
                pers_no=int(pers_no[-1:])
        
        sessions=glob.glob("%s//*" %inst)
        for ses in sessions:
            ses_no=ses[-6:]
            if ses_no[0]=="/":
                ses_no=int(ses_no[1])
            else:
                ses_no=int(ses_no[:2]) 
            
            if ses_no <= 5:
                training_input[tr_counter]=cv2.imread(ses,cv2.IMREAD_GRAYSCALE)
                training_class[tr_counter]=pers_no
                tr_counter=tr_counter+1
            else:
                test_input[test_counter]=cv2.imread(ses,cv2.IMREAD_GRAYSCALE)
                test_class[test_counter]=pers_no
                test_counter=test_counter+1        
  

    """   
    plt.imshow(training_input[0],cmap="gray")
    plt.imshow(training_input[1],cmap="gray") 
    plt.imshow(training_input[2],cmap="gray") 
    plt.imshow(training_input[3],cmap="gray") 
    plt.imshow(training_input[4],cmap="gray")  
    plt.imshow(test_input[0],cmap="gray")
    plt.imshow(test_input[1],cmap="gray") 
    plt.imshow(test_input[2],cmap="gray") 
    plt.imshow(test_input[3],cmap="gray") 
    plt.imshow(test_input[4],cmap="gray")  
    """
    np_training_input=np.array(training_input)
    np_test_input=np.array(test_input)
    np_training_class=np.array(training_class)
    np_test_class=np.array(test_class)
    #plt.imshow(np_training_input[2,],cmap="gray")
    return np_training_input,np_test_input,np_training_class,np_test_class



    