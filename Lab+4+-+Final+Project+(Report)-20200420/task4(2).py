# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:43:14 2020

@author: Tisana
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import lab4_hmc as hmc
from scipy.special import gamma
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


#standardising input x
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#add bias
x_train=np.concatenate((x_train,np.ones(x_train.shape[0])[:,np.newaxis]),axis=1)
x_test=np.concatenate((x_test,np.ones(x_test.shape[0])[:,np.newaxis]),axis=1)

#parameter
a0=10**(-2)
b0=10**(-4)

######################################
######################################
######################################
#check gradient
def energy_function(x0,f):
    x=f[0]
    y=f[1]
    s2=x0[0]
    alph=x0[1]
    w=x0[2:]
    
    #common term
    N=x.shape[0]
    M=x.shape[1]
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
 
######################################
######################################
######################################
#check energy function
'''
x0 = np.random.normal(size=11)
x0[0]=np.exp(x0[0])
x0[1]=np.exp(x0[1])
f=[x_train,y_train]
x=f[0]
y=f[1]
s2=x0[0]
alph=x0[1]
w=x0[2:]

#a
a_real=stats.multivariate_normal.logpdf(y,mean=np.matmul(x,w),cov=s2)   
a=energy_function(x0,f)[0]
print("check a : ",np.isclose(a,a_real)) 
print(a,a_real,abs(a-a_real))

#b
b_real=stats.multivariate_normal.logpdf(w,cov=(alph**-1)*np.identity(w.shape[0])) 
b=energy_function(x0,f)[1]
print("check b : ",np.isclose(b,b_real))
print(b,b_real,abs(b_real-b))
#c
c_real=stats.gamma.logpdf(1/alph,a=a0,scale=1/b0)
c=energy_function(x0,f)[2]
print("check c : ",np.isclose(c,c_real))
print(c,c_real,abs(c_real-c))
#d
d_real=stats.gamma.logpdf(s2,a=a0,scale=1/b0)
d=energy_function(x0,f)[3]
print("check d : ",np.isclose(d,d_real))
print(d,d_real,abs(d_real-d))
'''
######################################
######################################
######################################
def grad_function(x0,f):
    x=f[0]
    y=f[1]
    s2=x0[0]
    alph=x0[1]
    w=x0[2:]
    
    #common term
    N=x.shape[0]
    M=x.shape[1]    
    output_w=np.array([0]*len(w))
    output_s2=0
    output_alph=0
    
    #a wrt to w
    a_w=-2*np.matmul((y-np.matmul(x,w)),-x)   
    output_w=output_w-(a_w/(2*s2))
    
    #a wrt alph
    a_alph=0
    output_alph-=a_alph
    
    #a wrt s2
    term=(sum((y-np.matmul(x,w))**2))*(-(s2**-2))
    a_s2=(-0.5*N/s2)-(term/2)
    output_s2-=a_s2
    
    
    #b wrt w
    b_w=-alph*2*w/2
    output_w=output_w-b_w
    
    #b wrt alph
    b_alph=(M/(2*alph))-sum((w**2))/2
    output_alph-=b_alph
    
    #b wrt s2
    b_s2=0
    output_s2-=b_s2
  
    
    #c wrt w

    
    #c wrt alph
    c_alph=(-(a0-1)/alph)+(b0/(alph**2))
    output_alph-=c_alph
    
    #c wrt s2

   
    #d wrt w
    
    #d wrt alph
    
    #d wrt s2
    d_s2=((a0-1)/(s2))-b0
    output_s2-=d_s2
    

    return np.array([output_s2,output_alph]+output_w.tolist())


######################################
######################################
######################################
#check gradient
x0 = np.random.normal(size=11)
x0[0]=np.exp(x0[0])
x0[1]=np.exp(x0[1])
##x0=np.array([5.0,5.0])
print(x0)
f=[x_train,y_train]
hmc.gradient_check(x0, energy_function, grad_function, f)
######################################
######################################
######################################


np.random.seed(seed=1)  # For reproducibility 
eps=0.0035
R = 10000
burn = int(R/10)  
L = 100  
x0 = np.random.normal(size=11)
x0[0]=np.exp(x0[0])
x0[1]=np.exp(x0[1])

S, reject = hmc.sample(x0, energy_function, grad_function, R, L, eps, burn=burn, checkgrad=True, args=[f])
######################################
######################################
######################################
#plot to see convergence of estimate avlue
s2_list=S[:,0]
alph_list=S[:,1]
w_list=S[:,2:]
num=np.arange(0,s2_list.shape[0])

s2_list=np.cumsum(s2_list)/num
alph_list=np.cumsum(alph_list)/num
w1_list=np.cumsum(w_list[:,0])/num

plt.figure()
plt.title("convergence of s2")
plt.plot(num,s2_list)

plt.figure()
plt.title("convergence of alph")
plt.plot(num,alph_list)

plt.figure()
plt.title("convergence of w1")
plt.plot(num,w1_list)

######################################
######################################
######################################
#see RMSE

def RMSE_func(actual,pred):
    #sqrt_diff=np.mean((actual-pred)**2)
    sqrt_diff=sum((actual-pred)**2)
    sqrt_diff=sqrt_diff/actual.shape[0]
    return np.sqrt(sqrt_diff)
final_alph=alph_list[-1]
final_s2=s2_list[-1]
final_w=w_list.sum(axis=0)/10000
print("best s2 : ",final_s2)
print("best alph : ",final_alph)
print("best w : ",final_w)
y_pred_train=np.matmul(x_train,final_w)
y_pred_test=np.matmul(x_test,final_w)

RMSE_train=RMSE_func(y_train,y_pred_train)
RMSE_test=RMSE_func(y_test,y_pred_test)

print("RMSE train : ",RMSE_train)
print("RMSE test : ",RMSE_test)

######################################
######################################
######################################
#plot posterior
def log_posterior(x0,f):
    log_p=-energy_function(x0,f)
    return log_p

log_posterior_list=[]
i=0
for alph in alph_list[-100:]:
    i+=1
    print(i)
    p_s2=[]
    for s2 in s2_list[-100:]:

        
        x0=[s2,alph]+final_w.tolist()
        x0=np.array(x0)
        log_prob=log_posterior(x0,[x_train,y_train])
      
        p_s2.append(log_prob)


    log_posterior_list.append(np.array(p_s2))
log_posterior_list=np.array(log_posterior_list)


plt.figure()
plt.title("log posterior lnP(w,α,s2|y)")
plt.contourf(s2_list[-100:],alph_list[-100:],log_posterior_list)
plt.colorbar()
plt.scatter(final_s2,final_alph,c="red")
plt.xlabel(" s2")
plt.ylabel(" alph")