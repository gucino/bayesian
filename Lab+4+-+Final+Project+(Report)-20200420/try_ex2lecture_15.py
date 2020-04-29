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
for n in range(1,x_data.shape[1]):
    x=x_data[:,:n]
    y=y_data[:n]
    #update q w
    m_n,s_n=update_q_w(beta,x,y,a_n,b_n)
    w=stats.multivariate_normal.rvs(mean=np.squeeze(m_n),cov=s_n)
    q_w=stats.multivariate_normal.pdf(w,mean=np.squeeze(m_n),cov=s_n,allow_singular=True)
    
    #update q alpha
    a_n,b_n=update_q_alpha(a_0,M,b_0,m_n,s_n)
    alpha=stats.gamma.rvs(a_n,b_n)
    q_alpha=(stats.gamma.pdf(alpha,a_n,b_n)).sum()
    
    #total prob
    log_p=q_w*q_alpha
    print(log_p,w)
    l.append(log_p)

plt.figure()
plt.plot(np.arange(0,len(l)),l)    



y_pred=sum(w[:,np.newaxis]*x_data)
plt.figure()
plt.plot(y,y,c="red")
plt.plot(y_pred,y_pred,c="blue")