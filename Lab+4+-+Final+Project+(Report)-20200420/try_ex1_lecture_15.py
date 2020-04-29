# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:04:53 2020

@author: Tisana
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
real_mu=0
real_t=1
data=stats.multivariate_normal.rvs(mean=real_mu,cov=real_t,size=10000)

######################################
######################################
######################################
def update_q_mu(lam_0,mu_0,n,x_bar,a_n,b_n):
    mean=((lam_0*mu_0)+(n*x_bar))/(lam_0+n)
    var=(lam_0+n)*(a_n/b_n)
    return mean,var
def value_function(x,mu,lam_0,mu_0):
    value=sum((np.array(x)-mu)**2)+(lam_0*((mu-mu_0)**2))
    return value
def update_q_t(a_0,n,b_0,prob_mu,x,mu,mu_0,lam_0):
    a_n=a_0+(n/2)
    value=value_function(x,mu,lam_0,mu_0)
    b_n=b_0+0.5*prob_mu*(value)
    
    return a_n,b_n

######################################
######################################
######################################
mu_0=1
lam_0=1
a_0=1
b_0=1
######################################
######################################
######################################

a_n=a_0
b_n=b_0
#for each iteration
l=[]
mu_list=[]
t_list=[]
an_list=[]
b_n_list=[]
for n in range(1,len(data)):
    #update q mu
    x_bar=np.mean(data[:n])
    mean,var=update_q_mu(lam_0,mu_0,n,x_bar,a_n,b_n)
    
    #update q t
    mu=stats.multivariate_normal.rvs(mean=mean,cov=var)
    value=value_function(data,mu,lam_0,mu_0)
    prob_mu=stats.multivariate_normal.pdf(value,mean=mean,cov=var)
    x=data[:n]
    a_n,b_n=update_q_t(a_0,n,b_0,prob_mu,x,mu,mu_0,lam_0)
    
    #see probability
    prob_q_mu=stats.multivariate_normal.pdf(value,mean=mean,cov=var)
    t=stats.gamma.rvs(a_n,b_n)
    prob_q_t=stats.gamma.pdf(t,a_n,b_n)
    proposal_dist=prob_q_mu*prob_q_t
    
    print(proposal_dist)
    l.append(proposal_dist)
    mu_list.append(mu)
    t_list.append(t)
    an_list.append(a_n)
    b_n_list.append(b_n)
   
plt.figure()  
plt.plot(np.arange(0,len(l)),l)

plt.figure()  
plt.plot(np.arange(0,len(mu_list)),mu_list)

plt.figure()  
plt.plot(np.arange(0,len(t_list)),t_list)


plt.figure()  
plt.plot(np.arange(0,len(an_list)),an_list)

plt.figure()  
plt.plot(np.arange(0,len(b_n_list)),b_n_list)
