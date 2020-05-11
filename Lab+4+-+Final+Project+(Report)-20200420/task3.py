# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:12:45 2020

@author: Tisana
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import lab4_hmc as hmc

mean=np.array([0,0])
correlation=0.9
cov=np.array([[1,correlation],[correlation,1]])

x1=np.linspace(-3,3,500)
x2=np.linspace(-3,3,500)
x1,x2=np.meshgrid(x1,x2)
a=np.array([x1,x2])
a=a.T

prob=stats.multivariate_normal.pdf(a,mean,cov)
plt.figure()
plt.title("2D correlated Gaussian")
plt.contourf(x1,x2,prob)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()



######################################
######################################
######################################
#energy function
def energy_function(x,f):
    x0=x[0]
    x1=x[1]
    
    
    a=-np.log(2*np.pi)
    b=np.log((1-(correlation**2)))/2
    c=((-x0**2)+(2*correlation*x0*x1)-(x1**2))/(2*(1-(correlation**2)))
    log_p=a-b+c
    return -log_p

#grad
def grad_funtion(x,f):
    x0=x[0]
    x1=x[1]
    
    grad1=((2*x0)-(2*correlation*x1))/(2*(1-(correlation**2)))
    grad2=((2*x1)-(2*correlation*x0))/(2*(1-(correlation**2)))
    
    return np.array([grad1,grad2])
######################################
######################################
######################################
#validate energy
x0 = np.random.normal(size=2)
real_prob=stats.multivariate_normal.pdf(x0,mean,cov)
prob=np.exp(-(energy_function(x0,0)))
print("real : ",real_prob)
print("calculated : ",prob)
print("difference : ",abs(real_prob-prob))
######################################
######################################
######################################
#check gradient
f=0
x0 = np.random.normal(size=2)
grad_funtion(x0,f)
hmc.gradient_check(x0, energy_function, grad_funtion, f)

######################################
######################################
######################################
#find optimal epsilon step 1


#
np.random.seed(seed=1)  
R = 10000 
burn = int(R/10)  
L = 25  



eps = 0 
plt.figure()
plt.suptitle("tuning epsilon0")
for i in range(0,10):
    plt.subplot(5,2,i+1)
    
    eps=round(eps,3)
    S, reject = hmc.sample(x0, energy_function, grad_funtion, R, L, eps, burn=burn, checkgrad=True, args=[f])
    
    accept=round(100-(reject/R*100),3)
    title="epsilon : "+str(eps)+" // accept : "+str(accept)+"%"
    plt.plot(S[:, 0], S[:, 1], '.', ms=6, color='CadetBlue', alpha=0.25, zorder=0,label=title)
    plt.legend()
    plt.contour(x1, x2, prob, cmap='Reds', linewidths=3, zorder=1)
    eps+=0.1

######################################
######################################
######################################
#find optimal epsilon step 2
np.random.seed(seed=1)  
eps = 0.4
plt.figure()
plt.suptitle("tuning epsilon0")
for i in range(0,10):
    eps+=0.01
    plt.subplot(5,2,i+1)
    
    eps=round(eps,3)
    S, reject = hmc.sample(x0, energy_function, grad_funtion, R, L, eps, burn=burn, checkgrad=True, args=[f])
    
    accept=round(100-(reject/R*100),3)
    title="epsilon : "+str(eps)+" // accept : "+str(accept)+"%"
    plt.plot(S[:, 0], S[:, 1], '.', ms=6, color='CadetBlue', alpha=0.25, zorder=0,label=title)
    plt.legend()
    plt.contour(x1, x2, prob, cmap='Reds', linewidths=3, zorder=1)
    
    
######################################
######################################
######################################
#find optimal epsilon step 3
np.random.seed(seed=1)  
plt.figure()
plt.suptitle("tuning epsilon0")
eps_list=np.linspace(0.48,0.5,10)
for i in range(0,10):
    eps=eps_list[i]
    plt.subplot(5,2,i+1)
    
    eps=round(eps,3)
    S, reject = hmc.sample(x0, energy_function, grad_funtion, R, L, eps, burn=burn, checkgrad=True, args=[f])
    
    accept=round(100-(reject/R*100),3)
    title="epsilon : "+str(eps)+" // accept : "+str(accept)+"%"
    plt.plot(S[:, 0], S[:, 1], '.', ms=6, color='CadetBlue', alpha=0.25, zorder=0,label=title)
    plt.legend()
    plt.contour(x1, x2, prob, cmap='Reds', linewidths=3, zorder=1)
    
######################################
######################################
######################################
eps=0.487
S, reject = hmc.sample(x0, energy_function, grad_funtion, R, L, eps, burn=burn, checkgrad=True, args=[f])
plt.figure()
plt.plot(S[:, 0], S[:, 1], '.', ms=6, color='CadetBlue', alpha=0.25, zorder=0)    
plt.contour(x1, x2, prob, cmap='Reds', linewidths=3, zorder=1)

######################################
######################################
######################################
#plot to see value converge to mean of 0
num=np.arange(1,S.shape[0]+1)
value_x1=np.cumsum(S[:,0])/num
value_x2=np.cumsum(S[:,1])/num

plt.figure()
plt.title("show convergence of x1")
plt.plot(num,value_x1)

plt.figure()
plt.title("show convergence of x2")
plt.plot(num,value_x2)