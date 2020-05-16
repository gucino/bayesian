# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:59:35 2020

@author: Tisana
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


######################################
######################################
######################################
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

####################################################################
####################################################################
####################################################################
#task 1
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


######################################
######################################
######################################
#exploratory on each variable
plt.figure()
plt.suptitle("correlation between each attribute x and target value y")
for i in range(0,x_train.shape[1]):
    plt.subplot(2,4,i+1)
    title=attribute_x_list[i]
    plt.title(title)
    
    x=x_train[:,i]
    y=y_train
    
    corelation=np.corrcoef(x, y)
    
    plt.scatter(x,y,label=corelation)
    plt.legend()
plt.show()
######################################
######################################
######################################
#perform linear regression as baseline predictor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

#learn
regressor.fit(x_train,y_train)

#see train ans test result
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)

#compute error
train_error=np.sqrt(np.mean((y_train_pred-y_train)**2))
test_error=np.sqrt(np.mean((y_test_pred-y_test)**2))
print("train error : ",train_error)
print("test error : ",test_error)

c_train="correlation : "+str(np.corrcoef(y_train_pred, y_train)[0][1])
c_test="correlation : "+str(np.corrcoef(y_test, y_test_pred)[0][1])
print("correlation train : ", np.corrcoef(y_train_pred, y_train))
print("correlation test : ", np.corrcoef(y_test, y_test_pred))

#plot
plt.figure()
plt.suptitle("correlation between actual y and pedicted y (good prediction)")
plt.subplot(1,2,1)
plt.title("train")
plt.scatter(y_train,y_train_pred,label=c_train,c="red")
plt.xlabel("actual")
plt.ylabel("predict")
#plt.plot(y_train_pred,y_train_pred,label="predict",c="blue")
plt.legend()
plt.subplot(1,2,2)
plt.title("test")
plt.scatter(y_test,y_test_pred,label=c_test,c="red")
#plt.plot(y_test_pred,y_test_pred,label="predict",c="blue")
plt.legend()
plt.xlabel("actual")
plt.ylabel("predict")

######################################
######################################
######################################
#perform backward elimination
import statsmodels.api as sm
x=np.append(np.ones((x_train.shape[0],1)).astype(int),x_train,axis=1)

x_optimal=x[:,[0,1,2,3,4,5,6,7,8]]
regressor=sm.OLS(endog=y_train,exog=x_optimal).fit()
regressor.summary()

#remove first variable
x_optimal=x[:,[0,1,2,3,4,5,7,8]]
regressor=sm.OLS(endog=y_train,exog=x_optimal).fit()
regressor.summary()

#remove second variable
x_optimal=x[:,[0,1,2,3,4,5,7]]
regressor=sm.OLS(endog=y_train,exog=x_optimal).fit()
regressor.summary()   

#perform regression with optimal x
regressor=LinearRegression()

#learn
regressor.fit(x_optimal,y_train)

#see train ans test result
x=np.append(np.ones((x_test.shape[0],1)).astype(int),x_test,axis=1)
x_optimal_test=x[:,[0,1,2,3,4,5,7]]
y_train_pred=regressor.predict(x_optimal)
y_test_pred=regressor.predict(x_optimal_test)

#compute error
train_error=np.sqrt(np.mean((y_train_pred-y_train)**2))
test_error=np.sqrt(np.mean((y_test_pred-y_test)**2))
print("train error : ",train_error)
print("test error : ",test_error)
####################################################################
####################################################################
####################################################################
