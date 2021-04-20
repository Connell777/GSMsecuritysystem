#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
import scipy.fftpack
import math
from MPU6050 import run


def feat(dat):
    send_d = []
    tap = run(send_d)
    t=0
    x=0
    count=0
    countmag = 0
    for c in range(len(tap)):
        if(tap['Magnitude'].mean()>15100):
            globals()["mean"+str(t)]=1
        else:
            globals()["mean"+str(t)]=0
        if(tap['Magnitude'].std()>3500):
            globals()["std"+str(t)]=1
        else:
            globals()["std"+str(t)]=0         
        t=t+1
        if (count<334):
            count= count + 1
            if (tap['Magnitude'][c]>30000):
                countmag = countmag+1
        else:
            if(countmag<3):        
                globals()["thresh"+str(x)] = "Low"
            if(countmag>3 and countmag<6):
                globals()["thresh"+str(x)] = "Medium"
            if(countmag>=6):
                globals()["thresh"+str(x)] = "High"
            count = 0
            countmag=0
            
    s=0
    final = pd.DataFrame(columns=['mean','std','thresh'])
    for c in range(1):        
        final.loc[s] = [globals()["mean"+str(s)]] + [globals()["std"+str(s)]] + [globals()["thresh"+str(s)]] 
        s=s+1
    return final








