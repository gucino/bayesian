# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:42:16 2020

@author: Tisana
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#task 2
#import dataset
train_dataset=pd.read_csv("ee-train.csv",header=None).values
attribute_x_list=train_dataset[0,:-1].tolist()
traget_y=train_dataset[0,-1]

train_dataset=np.array(train_dataset[1:,:],dtype=np.float64)
x_train=train_dataset[:,:-1]
y_train=train_dataset[:,-1]

test_dataset=pd.read_csv("ee-test.csv").values
x_test=test_dataset[:,:-1]
y_test=test_dataset[:,-1]



#standardising input x
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#check mean of zero and std 1
print(np.isclose(np.mean(x_train,axis=0),[0]*x_train.shape[1]).all())
print(np.isclose(np.mean(x_test,axis=0),[0]*x_test.shape[1]).all())
print(np.isclose(np.std(x_train,axis=0),[1]*x_train.shape[1]).all())
print(np.isclose(np.std(x_test,axis=0),[1]*x_test.shape[1]).all())

#add bias to input x
x_train=np.concatenate((x_train,np.ones(x_train.shape[0])[:,np.newaxis]),axis=1)
x_test=np.concatenate((x_test,np.ones(x_test.shape[0])[:,np.newaxis]),axis=1)

######################################
######################################
######################################