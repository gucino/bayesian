# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:18:06 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:18:32 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:04:53 2020

@author: Tisana
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

real_w=np.array([1.5,3.8])
real_w=real_w[:,np.newaxis]
real_alpha=1
x1=np.linspace(1,10,10000000)
x2=np.linspace(1,10,10000000)
x_data=np.array([x1,x2])
y_data=sum(real_w*x_data)

######################################
######################################
######################################


train_dataset=pd.read_csv("ee-train.csv",header=None).values
attribute_x_list=train_dataset[0,:-1].tolist()
traget_y=train_dataset[0,-1]

train_dataset=np.array(train_dataset[1:,:],dtype=np.float64)
x_train=train_dataset[:,:-1]
x_train=x_train.T
y_train=train_dataset[:,-1]

test_dataset=pd.read_csv("ee-test.csv").values
x_test=test_dataset[:,:-1]
x_test=x_test.T
y_test=test_dataset[:,-1]
######################################
######################################
######################################
def update_q_alpha(a_0,M,b_0,m_n,s_n):
    a_n=a_0+(M/2)
    value=np.matmul(m_n,m_n.T)+s_n
    b_n=b_0+0.5*value
    return a_n,b_n

def update_q_w(beta,x,y,a_n,b_n):
    s_n=np.linalg.inv(((a_n/b_n)*np.identity(x.shape[0]))+(beta*np.matmul(x,x.T)))
    #m_n=beta*np.matmul(s_n,np.matmul(y[np.newaxis,:],x.T).T)
    m_n=beta*np.matmul(s_n,np.matmul(y[np.newaxis,:],x.T).T)
    return m_n,s_n
######################################
######################################
######################################
a_0=1
b_0=1
M=x_data.shape[0]
beta=1
######################################
######################################
######################################
a_n=a_0
b_n=b_0
l=[]
rmse__test_list=[]
rmse_train_list=[]
for n in range(1,x_train.shape[1]*3):
    x=x_train[:,:n]
    y=y_train[:n]
    #update q w
    m_n,s_n=update_q_w(beta,x,y,a_n,b_n)
    #w=stats.multivariate_normal.rvs(mean=np.squeeze(m_n),cov=s_n)
    w=np.squeeze(m_n)
    q_w=stats.multivariate_normal.pdf(w,mean=np.squeeze(m_n),cov=s_n,allow_singular=True)
    
    #update q alpha
    a_n,b_n=update_q_alpha(a_0,M,b_0,m_n,s_n)
    #alpha=stats.gamma.rvs(a=a_n,loc=b_n)
    alpha=a_n
    q_alpha=(stats.gamma.pdf(x=alpha,a=a_n,loc=b_n)).sum()
    
    #total prob
    log_p=q_w*q_alpha
    #print(log_p,w)
    l.append(log_p)
    
    #pred
    y_pred_test=sum(w[:,np.newaxis]*x_test)
    rmse_test=np.sqrt(np.mean((y_test-y_pred_test)**2))
    rmse__test_list.append(rmse_test)
    
    y_pred_train=sum(w[:,np.newaxis]*x_train)
    rmse_train=np.sqrt(np.mean((y_train-y_pred_train)**2))
    rmse_train_list.append(rmse_train)
    print(alpha)

plt.figure()
plt.plot(np.arange(0,len(l)),l)    

plt.figure()
plt.plot(np.arange(0,len(rmse_train_list)),rmse_train_list,label="train error")
plt.plot(np.arange(0,len(rmse__test_list)),rmse__test_list,label="train error")

print("RMSE train : ",rmse_train_list[-1])
print("RMSE test : ",rmse__test_list[-1])