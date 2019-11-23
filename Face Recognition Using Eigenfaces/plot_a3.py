#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:59:27 2019

@author: can
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import cv2

def plot_eigen_faces(U_matrix,which_eigen_vector,width,height):
    selected_eigen_vector=U_matrix[:,which_eigen_vector]
    selected_eigen_vector=np.transpose(selected_eigen_vector)
    selected_eigen_vector=selected_eigen_vector.reshape(height,width)
    #plt.imshow(selected_eigen_vector,cmap="gray")
    return selected_eigen_vector
    
    
def eigen_face_projection(U,selected_image,average_face):
    tmp_image=selected_image-average_face
    projected_image=np.matmul(np.transpose(U),tmp_image)
    return projected_image

def draw_picture_with_n_features(projected_features,U,average_face,width,height,how_many_vec):
    diag_features=np.diag(projected_features,0)
    projected_pic=np.matmul(U,diag_features)
    projected_pic=projected_pic[:,:how_many_vec]
    projected_pic=np.sum(projected_pic,axis=1)
    projected_pic=projected_pic+average_face
    #plt.imshow((np.transpose(projected_pic)).reshape(height,width),cmap="gray")
    return (np.transpose(projected_pic)).reshape(height,width)

