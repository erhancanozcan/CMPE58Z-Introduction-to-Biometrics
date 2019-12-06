#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:51:30 2019

@author: can
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:55:00 2019

@author: can
"""

#%%
#s='355879ACB6'
#s[:4] + '-' + s[4:]

import glob
import numpy as np
from shutil import copyfile
import os
where_is_the_folder="/Users/can/Desktop/biometrics_proje_scripts" # this must be adjusted

def fill_train_samples_all_emotions(where_is_the_folder):
    
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    address_partcpnts=where_is_the_folder + "/ck+/Emotion//*"
    address_pics=where_is_the_folder + "/ck+/cohn-kanade-images"
    empty_file=where_is_the_folder+"/train_samples_all_emotions"
    participants = glob.glob(address_partcpnts)  #Returns a list of all folders
    #participants = glob.glob("/Users/can/Documents/Biometrics/project/ck+/Emotion//*") #Returns a list of all folders
    
    
    
    #x=participants[6]
    #sessions=glob.glob("%s//*" %x)[1]
    #files=glob.glob("%s//*" %sessions)[0]
    
    counter=1
    fold_name="ins"+str(counter)
    minn=100
    maxx=0
    
    for x in participants:
        part = "%s" %x[-4:] #store current participant number
        for sessions in glob.glob("%s//*" %x): #Store list of sessions for current participant
            for files in glob.glob("%s//*" %sessions):
                current_session = sessions[-3:]
                #current_session = files
                file = open(files, 'r')
    
                emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
    
                #pics=glob.glob("/Users/can/Documents/Biometrics/project/ck+/cohn-kanade-images//%s//%s//*" %(part, current_session))
                pics=glob.glob(address_pics+"//%s//%s//*" %(part,current_session))
                
                #selected=pics[0]
                for selected in pics:
                    
                    selected_no=selected[-6:]
                    selected_no=int(selected_no[:2])
                    if selected_no <= minn:
                        neut_source=selected
                        minn=selected_no
                    if selected_no >= maxx:
                        emo_source=selected
                        maxx=selected_no
                
                minn=100
                maxx=0
                #sourcefile_emotion = glob.glob("/Users/can/Documents/Biometrics/project/ck+/cohn-kanade-images//%s//%s//*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
                
                #sourcefile_neutral = glob.glob("/Users/can/Documents/Biometrics/project/ck+/cohn-kanade-images//%s//%s//*" %(part, current_session))[0] #do same for neutral image
                #print sourcefile_neutral
    
                
                #os.chdir("/Users/can/Desktop/Bio_CK+")
                os.chdir(empty_file)
                path = "./" + fold_name
                os.mkdir(path)
                
                #os.getcwd()
                #glob.glob("/Users/can/Desktop/Bio_CK+//*")
                
                #dest_neut = "/Users/can/Desktop/Bio_CK+//%s//%s" %(path,neut_source[-21:]) #Generate path to put neutral image
                dest_neut = (empty_file+"//%s//%s" %(path,neut_source[-21:])) #Generate path to put neutral image
                
                dest_emot = (empty_file+"//%s//%s" %(path,emo_source[-21:]))
                
                emotion_name=fold_name+files[-11:]
                dest_files= (empty_file+"//%s//%s" %(path,emotion_name))
                
                copyfile(neut_source, dest_neut) #Copy file
                copyfile(emo_source, dest_emot) #Copy file
                copyfile(files,dest_files)
                
                counter=counter+1
                fold_name="ins"+str(counter)
#%%
