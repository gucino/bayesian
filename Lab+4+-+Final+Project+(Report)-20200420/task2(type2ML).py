# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:20:06 2020

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
#marginal likelihood
def log_marginal_likelihood(x,y,alph,s2):
    cov=s2*np.identity(x.shape[0])+np.matmul((alph**-1)*x,x.T)

    log_p=stats.multivariate_normal.logpdf(y,cov=cov,allow_singular=True) 
    return log_p


#perform type 2 ML
best_log_alpha=0
best_log_s2=0
log_alpha_list=np.linspace(-25,-5,100)
log_s2_list=np.linspace(-10,10,100)

log_list=[]
max_lgp=-999999
i=0
for log_s2 in log_s2_list:
    i+=1
    print(i)
    row_list=[]
    for log_alpha in log_alpha_list:
        alph=np.exp(log_alpha)
        s2=np.exp(log_s2)   

        lgp=log_marginal_likelihood(x_train,y_train,alph,s2)
        #print(lgp)
        row_list.append(lgp)
        if lgp>max_lgp:
            max_lgp=lgp
            best_log_alpha=log_alpha
            best_log_s2=log_s2
    log_list.append(np.array(row_list))
log_list=np.array(log_list)

best_alph=np.exp(best_log_alpha)
best_s2=np.exp(best_log_s2)
print("max log likeihood ",max_lgp) 
print("best log alpha ",best_log_alpha," | best alpha : ",best_alph)
print("best log s2 ",best_log_s2," | best s2 : ",best_s2)


log_alpha_list,log_s2_list=np.meshgrid(log_alpha_list,log_s2_list)
plt.figure()
plt.title("ln(p(y|α,s2))")
plt.contourf(log_s2_list,log_alpha_list,log_list) 
plt.scatter(best_log_s2,best_log_alpha,c="red")
plt.xlabel("log alpha")
plt.ylabel("log s2") 

plt.figure()
plt.title("p(y|α,s2)")
plt.contourf(log_s2_list,log_alpha_list,np.exp(log_list) )
plt.scatter(best_log_s2,best_log_alpha,c="red")
plt.xlabel("log alpha")
plt.ylabel("log s2") 

######################################
######################################
######################################
#see the affect of regularization parameter

#prediction
def weight(x, y, alph,s2):
    #
    #### **** YOUR CODE HERE **** ####
    #
    lam=alph*s2
    
    transpose_x=np.transpose(x)
    
    a=np.matmul(transpose_x,x)
    b=lam*np.identity(a.shape[0])
    c=np.matmul(transpose_x,y)
    
    mean=np.matmul(np.linalg.inv(a+b),c)
    mean=np.squeeze(mean)
    
    var=s2*np.linalg.inv(a+b)
    return mean,var

def prediction(x_train, y_train, alph,s2,x_test):
    w,_=weight(x_train, y_train, alph,s2)
    y_pred=np.matmul(x_test,w)
    return y_pred

def RMSE_func(actual,pred):
    #sqrt_diff=np.mean((actual-pred)**2)
    sqrt_diff=sum((actual-pred)**2)
    sqrt_diff=sqrt_diff/actual.shape[0]
    return np.sqrt(sqrt_diff)
#best case
y_pred_train=prediction(x_train, y_train, best_alph,best_s2,x_train)
y_pred_test=prediction(x_train, y_train, best_alph,best_s2,x_test)
RMSE_train=RMSE_func(y_pred_train,y_train)
RMSE_test=RMSE_func(y_pred_test,y_test)
print("RMSE train : ",RMSE_train)
print("RMSE test : ",RMSE_test)

#under fit case
y_pred_train=prediction(x_train, y_train, 10,10,x_train)
y_pred_test=prediction(x_train, y_train, 10,10,x_test)
RMSE_train=RMSE_func(y_pred_train,y_train)
RMSE_test=RMSE_func(y_pred_test,y_test)
print("RMSE train : ",RMSE_train)
print("RMSE test : ",RMSE_test)

#over fit case
y_pred_train=prediction(x_train, y_train, 0,0,x_train)
y_pred_test=prediction(x_train, y_train, 0,0,x_test)
RMSE_train=RMSE_func(y_pred_train,y_train)
RMSE_test=RMSE_func(y_pred_test,y_test)
print("RMSE train : ",RMSE_train)
print("RMSE test : ",RMSE_test)
