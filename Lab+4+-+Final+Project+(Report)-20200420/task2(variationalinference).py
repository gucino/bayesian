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
#note lambda : 1/s2
a_0_lambda=1
b_0_lambda=1
a_0_alph=1
b_0_alph=1

a_n_alph=a_0_alph
b_n_alph=b_0_alph

#update prob of w and lambda
def update_w_and_lambda(a_n_alph,b_n_alph,x,y):
    v_n=np.linalg.inv((a_n_alph/b_n_alph)*np.identity(x.shape[1])+np.matmul(x.T,x))
    w_n=np.squeeze(np.matmul(np.matmul(v_n,x.T),y[:,np.newaxis]))
    
    num_sample=x.shape[0]
    a_n_lambda=a_0_lambda+(num_sample/2)
    value1=np.linalg.norm(y-np.squeeze(np.matmul(x,w_n[:,np.newaxis])))**2
    value2=np.squeeze(np.matmul(w_n[:,np.newaxis].T,(a_n_alph/b_n_alph)*w_n[:,np.newaxis]))
    b_n_lambda=b_0_lambda+0.5*(value1+value2)
    
    #calculate prob
    w=w_n
    lam=a_n_lambda/b_n_lambda
    prob_w=stats.multivariate_normal.pdf(w,w_n,v_n/lam)

    prob_lambda=stats.gamma.pdf(x=lam,a=a_n_lambda,scale=1/b_n_lambda)
    return prob_w,w_n,v_n,prob_lambda,a_n_lambda,b_n_lambda

#update prob alph
def update_alph(a_0_alph,b_0_alph,w_n,v_n,a_n_lambda,b_n_lambda,x):
    d=x.shape[1]
    a_n_alph=a_0_alph+(d/2)
    value=np.squeeze(np.matmul((a_n_lambda/b_n_lambda)*w_n[:,np.newaxis].T,w_n[:,np.newaxis]))
    b_n_alph=b_0_alph+0.5*(value+np.trace(v_n))
    
    #calculate prob
    alph=a_n_alph/b_n_alph
    prob_alph=stats.gamma.pdf(x=alph,a=a_n_alph,scale=1/b_n_alph)
    return prob_alph,a_n_alph,b_n_alph

def prediction(x_train, y_train, alph,s2,x_test,w):
    y_pred=np.matmul(x_test,w)
    return y_pred

def RMSE_func(actual,pred):
    #sqrt_diff=np.mean((actual-pred)**2)
    sqrt_diff=sum((actual-pred)**2)
    sqrt_diff=sqrt_diff/actual.shape[0]
    return np.sqrt(sqrt_diff)

#start variational inference
prob_list=[]
log_p_list=[]

alpha_list=[]
s2_list=[]
RMSE_train_ist=[]
RMSE_test_list=[]
for i in range(1,x_train.shape[0]+1):
    #take sample
    x_sample=x_train[:i,:]
    y_sample=y_train[:i]
    
    #update prob of w and lambda
    prob_w,w_n,v_n,prob_lambda,a_n_lambda,b_n_lambda=\
    update_w_and_lambda(a_n_alph,b_n_alph,x_sample,y_sample)
    
    #update prob alph
    prob_alph,a_n_alph,b_n_alph=\
    update_alph(a_0_alph,b_0_alph,w_n,v_n,a_n_lambda,b_n_lambda,x_sample)
    
    #total pob
    total_prob=prob_w*prob_lambda*prob_alph
    #print(prob_w,prob_lambda,prob_alph)
    prob_list.append(total_prob)
    log_p_list.append(np.log(total_prob))
    
    #make prediction
    alph=a_n_alph/b_n_alph
    lam=a_n_lambda/b_n_lambda
    s2=1/lam
    y_pred_train=prediction(x_train, y_train, alph,s2,x_train,w_n)
    y_pred_test=prediction(x_train, y_train, alph,s2,x_test,w_n)
    alpha_list.append(alph)
    s2_list.append(s2)
    
    #calcualte error
    RMSE_train=RMSE_func(y_pred_train,y_train)
    RMSE_test=RMSE_func(y_pred_test,y_test)
    RMSE_train_ist.append(RMSE_train)
    RMSE_test_list.append(RMSE_test)
    
    print(i,RMSE_train,RMSE_test)
    

#plot to see result
plt.figure()
plt.title("error")
plt.plot(np.arange(0,len(RMSE_train_ist)),RMSE_train_ist,label="train RMSE")
plt.plot(np.arange(0,len(RMSE_test_list)),RMSE_test_list,label="test RMSE")
plt.legend()
plt.xlabel("num sample")
plt.ylabel("RMSE")
plt.show()

plt.figure()
plt.title("log prob")
plt.plot(np.arange(0,len(log_p_list)),log_p_list)
plt.xlabel("num sample")
plt.ylabel("log porb")
plt.show()



best_alph=alph
best_s2=s2
best_weight=w_n
print("best alph : ",best_alph)
print("best s2 : ",best_s2)
print("best weight : ",best_weight)
print("RMSE train : ",RMSE_train)
print("RMSE test : ",RMSE_test)

######################################
######################################
######################################
#plot posterior and log marginal likelihood
def energy_function(x0,ff):
    x=ff[0]
    y=ff[1]
    s2=x0[0]
    alph=x0[1]
    w=x0[2:]
    a0=10**(-2)
    b0=10**(-4)
    #common term
    N=x.shape[0]
    M=x.shape[1]
    from scipy.special import gamma
    term=(b0**a0)/gamma(a0)
    energy=0
    
    #a
  
    second_term=sum((y-np.matmul(x,w))**2)/(2*s2)
    a=-0.5*N*np.log(2*np.pi*s2)-second_term
    energy-=a
    
  
    #b
    b=0.5*M*np.log((alph/(2*np.pi)))-sum(alph*(w**2)/2)
    energy-=b
  
    #c
    c=np.log(term)-(a0-1)*np.log(alph)-(b0/alph)
    energy-=c
    
    #d
    d=np.log(term)+((a0-1)*np.log(s2))-(b0*s2)
    energy-=d

    #return a,b,c,d,energy
    return energy

def log_posterior(x0,f):
    log_p=-energy_function(x0,f)
    return log_p

alpha_list=np.linspace(0,1,100)
s2_list=np.linspace(0,60,100)
log_posterior_list=[]
i=0
for alph in alpha_list:
    i+=1
    print(i)
    p_s2=[]
    for s2 in s2_list:

        
        x0=[s2,alph]+best_weight.tolist()
        x0=np.array(x0)
        log_prob=log_posterior(x0,[x_train,y_train])
      
        p_s2.append(log_prob)


    log_posterior_list.append(np.array(p_s2))
log_posterior_list=np.array(log_posterior_list)


plt.figure()
plt.title("log posterior lnP(w,α,s2|y)")
plt.contourf(s2_list,alpha_list,log_posterior_list)
plt.colorbar()
plt.scatter(best_s2,best_alph,c="red")
plt.xlabel(" s2")
plt.ylabel(" alph")

