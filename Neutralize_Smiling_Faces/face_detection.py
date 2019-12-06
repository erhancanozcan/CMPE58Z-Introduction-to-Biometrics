#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:18:42 2019

@author: can
"""

import glob
import numpy as np
from shutil import copyfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import dlib
import face_recognition




def face_detect_allign_resize(folder_address):
    #for ck+ data set.
    face_cascade = cv2.CascadeClassifier('/anaconda2/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    #faces = face_cascade.detectMultiScale(img, 1.3, 3)
    
    
    width = 32
    height = 32
    dim = (width, height)
    #resized = cv2.resize(new_img, dim, interpolation = cv2.INTER_AREA)
    #plt.imshow(resized,cmap="gray")
    instances = glob.glob(folder_address+"/train_samples_all_emotions//*") #Returns a list of all folders
    i=0
    
    #inst=instances[0]
    #session=sessions[1]
    for inst in instances:
        sessions=glob.glob("%s//*" %inst)
        for session in sessions:
            ses_type=session[-3:]
            if ses_type!="txt":
                img = cv2.imread(session,cv2.IMREAD_GRAYSCALE)
                faces = face_cascade.detectMultiScale(img, 1.3, 5)
                if len(faces)==0:
                    faces = face_cascade.detectMultiScale(img, 1.3, 3)
                x=faces[0,0]
                y=faces[0,1]
                w=faces[0,2]
                h=faces[0,3]
                new_img=img[y:y+h,x:x+w]
                #plt.imshow(new_img,cmap="gray")
                
                #resized = cv2.resize(new_img, dim, interpolation = cv2.INTER_AREA)
                #plt.imshow(resized,cmap="gray")
                
                #up to here we detected the face, and cropped it. Now
                #we will use landmark detector from dlib. Then, we will
                #allign the face according to the positions of eyes.
                face_landmark_list=face_recognition.face_landmarks(new_img)
                desiredLeftEye=(0.30,0.30)
                desiredFaceWidth=258
                desiredFaceHeight=258
                
                
                ld_list=(face_landmark_list[0])
                leftEyePts=ld_list["left_eye"]
                leftEyePts=np.array(leftEyePts)
                rightEyePts=ld_list["right_eye"]
                rightEyePts=np.array(rightEyePts)
                
                # compute the center of mass for each eye
                leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
                rightEyeCenter = rightEyePts.mean(axis=0).astype("int")


                # compute the angle between the eye centroids
                #Angle is computed but it is not used in the
                #getRotationMatrix2D() function.
                dY = rightEyeCenter[1] - leftEyeCenter[1]
                dX = rightEyeCenter[0] - leftEyeCenter[0]
                angle = np.degrees(np.arctan2(dY, dX)) - 180
                
                
                # compute the desired right eye x-coordinate based on the
                # desired x-coordinate of the left eye
                desiredRightEyeX = 1.0 - desiredLeftEye[0]
 
                # determine the scale of the new resulting image by taking
                # the ratio of the distance between eyes in the *current*
                # image to the ratio of distance between eyes in the
                # *desired* image
                dist = np.sqrt((dX ** 2) + (dY ** 2))
                desiredDist = (desiredRightEyeX - desiredLeftEye[0])
                desiredDist *= desiredFaceWidth
                scale = desiredDist / dist
                
                
                # compute center (x, y)-coordinates (i.e., the median point)
                # between the two eyes in the input image
                eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
                # grab the rotation matrix for rotating and scaling the face
                M = cv2.getRotationMatrix2D(eyesCenter, 0, scale)
 
                # update the translation component of the matrix
                tX = desiredFaceWidth * 0.5
                tY = desiredFaceHeight * desiredLeftEye[1]
                M[0, 2] += (tX - eyesCenter[0])
                M[1, 2] += (tY - eyesCenter[1])

                # apply the affine transformation
                (w, h) = (desiredFaceWidth, desiredFaceHeight)
                output = cv2.warpAffine(new_img, M, (w, h))
                #plt.imshow(output,cmap="gray")
                
                #output = cv2.flip( output, -1 )
                #plt.imshow(output,cmap="gray")
                
                resized = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
                #plt.imshow(resized,cmap="gray")
                status = cv2.imwrite(session,resized)
        i=i+1
        
def detect_faces_and_resize_images(folder_address):
    

    #for ck+. But this function does not do allignment. So, it is not used.
    face_cascade = cv2.CascadeClassifier('/anaconda2/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    #faces = face_cascade.detectMultiScale(img, 1.3, 3)
    
    
    width = 32
    height = 32
    dim = (width, height)
    #resized = cv2.resize(new_img, dim, interpolation = cv2.INTER_AREA)
    #plt.imshow(resized,cmap="gray")
    instances = glob.glob(folder_address+"/train_samples_all_emotions//*") #Returns a list of all folders
    i=0
    for inst in instances:
        sessions=glob.glob("%s//*" %inst)
        for session in sessions:
            ses_type=session[-3:]
            if ses_type!="txt":
                img = cv2.imread(session,cv2.IMREAD_GRAYSCALE)
                faces = face_cascade.detectMultiScale(img, 1.3, 5)
                if len(faces)==0:
                    faces = face_cascade.detectMultiScale(img, 1.3, 3)
                x=faces[0,0]
                y=faces[0,1]
                w=faces[0,2]
                h=faces[0,3]
                new_img=img[y:y+h,x:x+w]
                resized = cv2.resize(new_img, dim, interpolation = cv2.INTER_AREA)
                status = cv2.imwrite(session,resized)
                
        #print i
        i=i+1

def obtain_flatten_images_of_desired_emotion(emotion_index,folder_address):
    #for ck+
    width = 32
    height = 32
    dim = (width, height)
    
    instances = glob.glob(folder_address+"/train_samples_all_emotions//*") #Returns a list of all folders
    i=0
    emotion_counter=np.zeros(8)
    
    for inst in instances:
        sessions=glob.glob("%s//*" %inst)
        for session in sessions:
            ses_type=session[-3:]
            if ses_type=="txt":
                file = open(session, 'r')
                emotion = int(float(file.readline()))
                emotion_counter[emotion]+=1        
        #print i
        i=i+1
        
    #emotion_counter #you can see the count of expressions.
    #emotion_index=5
    width_px=32
    height_px=32
    
    no_of_training_pairs=int(emotion_counter[emotion_index])
    neutral_pics=np.zeros((no_of_training_pairs,width_px,height_px))
    emotional_pics=np.zeros((no_of_training_pairs,width_px,height_px))
    
    focused_instances= [None]*no_of_training_pairs
    
    i=0
    for inst in instances:
        sessions=glob.glob("%s//*" %inst)
        for session in sessions:
            ses_type=session[-3:]
            if ses_type=="txt":
                file = open(session, 'r')
                emotion = int(float(file.readline()))
                if emotion == emotion_index:
                    focused_instances[i]=inst
                    i+=1
                
        #print i
        #i=i+1
    i=0
    for inst in focused_instances:
        sessions=glob.glob("%s//*" %inst)
        for session in sessions:
            ses_type=session[-3:]
            if ses_type!="txt":
                selected_no=session[-6:]
                selected_no=int(selected_no[:2])
                if selected_no==1:
                    neutral_pics[i]=cv2.imread(session,cv2.IMREAD_GRAYSCALE)
                else:
                    emotional_pics[i]=cv2.imread(session,cv2.IMREAD_GRAYSCALE)
                
        i=i+1    
    
    
    flat_emotional=emotional_pics.flatten()
    flat_neutral=neutral_pics.flatten()
    
    os.chdir(folder_address)
    os.getcwd()
    np.savetxt("flat_emotional.csv", flat_emotional, delimiter=",")
    np.savetxt("flat_neutral.csv", flat_neutral, delimiter=",")
 
    
def face_detect_allign_resize_wild_images(folder_address):
    #for ck+ data set.
    face_cascade = cv2.CascadeClassifier('/anaconda2/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    #faces = face_cascade.detectMultiScale(img, 1.3, 3)
    
    
    width = 32
    height = 32
    dim = (width, height)
    #resized = cv2.resize(new_img, dim, interpolation = cv2.INTER_AREA)
    #plt.imshow(resized,cmap="gray")
    instances = glob.glob(folder_address+"/wild_sample_images//*") #Returns a list of all folders
    i=0
    
    #inst=instances[0]
    for inst in instances:

            img = cv2.imread(inst,cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            if len(faces)==0:
                faces = face_cascade.detectMultiScale(img, 1.3, 3)
            x=faces[0,0]
            y=faces[0,1]
            w=faces[0,2]
            h=faces[0,3]
            new_img=img[y:y+h,x:x+w]
            #plt.imshow(new_img,cmap="gray")
            
            #resized = cv2.resize(new_img, dim, interpolation = cv2.INTER_AREA)
            #plt.imshow(resized,cmap="gray")
            
            #up to here we detected the face, and cropped it. Now
            #we will use landmark detector from dlib. Then, we will
            #allign the face according to the positions of eyes.
            face_landmark_list=face_recognition.face_landmarks(new_img)
            desiredLeftEye=(0.30,0.30)
            desiredFaceWidth=258
            desiredFaceHeight=258
            
            
            ld_list=(face_landmark_list[0])
            leftEyePts=ld_list["left_eye"]
            leftEyePts=np.array(leftEyePts)
            rightEyePts=ld_list["right_eye"]
            rightEyePts=np.array(rightEyePts)
            
            # compute the center of mass for each eye
            leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
            rightEyeCenter = rightEyePts.mean(axis=0).astype("int")


            # compute the angle between the eye centroids
            #Angle is computed but it is not used in the
            #getRotationMatrix2D() function.
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            
            
            # compute the desired right eye x-coordinate based on the
            # desired x-coordinate of the left eye
            desiredRightEyeX = 1.0 - desiredLeftEye[0]
 
            # determine the scale of the new resulting image by taking
            # the ratio of the distance between eyes in the *current*
            # image to the ratio of distance between eyes in the
            # *desired* image
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - desiredLeftEye[0])
            desiredDist *= desiredFaceWidth
            scale = desiredDist / dist
            
            
            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input image
            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, 0, scale)
 
            # update the translation component of the matrix
            tX = desiredFaceWidth * 0.5
            tY = desiredFaceHeight * desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])

            # apply the affine transformation
            (w, h) = (desiredFaceWidth, desiredFaceHeight)
            output = cv2.warpAffine(new_img, M, (w, h))
            #plt.imshow(output,cmap="gray")
            
            #output = cv2.flip( output, -1 )
            #plt.imshow(output,cmap="gray")
            
            resized = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
            #plt.imshow(resized,cmap="gray")
            new_address=folder_address+"/wild_sample_images_preprocessed/"+str(i)+".JPG"
            status = cv2.imwrite(new_address,resized)
            i=i+1
            
        
        
