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

####################################################################
####################################################################
####################################################################
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

def cov2_function(x0,f):
    x=f[0]
    y=f[1]
    s2=x0[0]
    alph=x0[1]
    
    b=x/alph
    b=np.dot(b,x.T)
    
    a=s2*np.identity(b.shape[0])    
    
    cov2=a+b
    return cov2


####################################################################
####################################################################
####################################################################
#check prob

#real value

'''
#check total
alph=5
s2=5
w,cov_w=weight(x_train, y_train, alph,s2)
log_p_w=(stats.multivariate_normal.logpdf(w,w,cov_w))
cov_alph=s2*np.identity(x_train.shape[0])+np.matmul((alph**-1)*x_train,x_train.T)
log_p_alph=stats.multivariate_normal.logpdf(x=y_train,cov=cov_alph)
real_log_p=log_p_alph+log_p_w


energy,_,_=energy_function(np.array([s2,alph]),[x_train,y_train])
log_p=-energy
print(np.isclose(log_p,real_log_p))
'''
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
    
    energy=0
    cov2=cov2_function(x0,f)
    mean_w,cov1=weight(x, y, alph,s2)
    N=x.shape[0]
    K=x.shape[1]
    inv_cov1=np.linalg.inv(cov1)

    #d
  
    sign,d=np.linalg.slogdet(cov2)
    energy+=d
    
    
    #e
    e=np.matmul(y.T,np.linalg.inv(cov2))
    e=np.matmul(e,y)
    energy+=e

    #a
    sign,a=np.linalg.slogdet(cov1)
    energy+=a    
 
    #c
    c=K*np.log(2*np.pi)
    energy+=c
    
    #f
    f=N*np.log(2*np.pi)
    energy+=f    
  
    #b
 
    b=np.matmul((w-mean_w).T,inv_cov1)
    b=np.matmul(b,(w-mean_w))
    energy+=b
 



    #return energy/2,log_w,log_alpha
    return energy/2

def grad_function(x0,f):
    x=f[0]
    y=f[1]
    s2=x0[0]
    alph=x0[1]
    w=x0[2:]
    
    output_s2=0
    output_alph=0
    output_w=[0]*len(w)
    
    cov2=cov2_function(x0,f)
    inv_cov2=np.linalg.inv(cov2)
    mu1,cov1=weight(x, y, alph,s2)
    cino=np.matmul(x.T,x)+(s2*alph*np.identity(x.shape[1]))
    inv_cino=np.linalg.inv(cino)
    determinant_cov1=np.linalg.det(cov1)
    #####################################
    #####################################
   
    #d wrt s2
    determinant_cov2=np.linalg.det(cov2)
    if np.isclose(determinant_cov2,0)==True:
        determinant_cov2=np.exp(np.log1p(determinant_cov2))

    d_s2=1/determinant_cov2
    adj_cov2=determinant_cov2*np.linalg.inv(cov2)
    d_s2*=np.trace(adj_cov2*np.identity(adj_cov2.shape[0]))
    output_s2+=d_s2/2
    
    #d wrt alph
    term=-(alph**-2)*x
    term=np.dot(term,x.T)
    d_alph=1/determinant_cov2
    d_alph*=np.trace(np.dot(adj_cov2,term))
    output_alph+=d_alph/2
    
    #d wrt w1
    d_w=[0]*len(w)
    output_w+=d_w
    
    
    #####################################
    #####################################
    
    #e wrt s2
    e_s2=np.matmul(y.T,inv_cov2*-1)
    e_s2=np.matmul(e_s2,np.identity(e_s2.shape[0]))
    e_s2=np.matmul(e_s2,inv_cov2)
    e_s2=np.matmul(e_s2,y)
    output_s2+=e_s2/2
    
    #e wrt alph
    neg_alph_power_minus_2=-alph**(-2)
    e_alph=neg_alph_power_minus_2*x
    e_alph=np.matmul(e_alph,x.T)
    e_alph=np.matmul(inv_cov2*-1,e_alph)
    e_alph=np.matmul(e_alph,inv_cov2)
    e_alph=np.matmul(y.T,e_alph)
    e_alph=np.matmul(e_alph,y)
    output_alph+=e_alph/2
    
    #e wrt w
    e_w=[0]*len(w)
    output_w+=e_w
    
   
    #####################################
    #####################################
    
    #a wrt s2
    if np.isclose(determinant_cov1,0)==True:
        determinant_cov1=np.exp(np.log1p(determinant_cov1))
       
    a_s2=1/determinant_cov1
    adj_cov1=determinant_cov1*np.linalg.inv(cov1)
    
    term=np.matmul(s2*-inv_cino,alph*np.identity(cino.shape[0]))
    term=np.matmul(term,inv_cino)+inv_cino
    
    a_s2*=np.trace(np.matmul(adj_cov1,term))
    output_s2+=a_s2   /2 
    
    #a wrt alph
    a_alph=1/determinant_cov1
    
    term=np.matmul(s2*-inv_cino,s2*np.identity(cino.shape[0]))
    term=np.matmul(term,inv_cino)
    
    a_alph*=np.trace(np.matmul(adj_cov1,term))
    output_alph+=a_alph /2  
    
    #a wrt w
    a_w=[0]*len(w)
    output_w+=a_w
    #####################################
    #####################################
   
    #b wrt s2
    output_s2+=b_wrt_s2(x, y, alph,s2,w)/2
    output_alph+=b_wrt_alph(x, y, alph,s2,w)/2
    output_w=(b_wrt_w(x, y, alph,s2,w)/2).tolist()
 
    return np.array([output_s2,output_alph]+output_w)

######################################
######################################
######################################
def b1_s2_func(x, y, alph,s2):
    b11=np.matmul(x.T,x)+(s2*alph*np.identity(x.shape[1]).T)
    output=np.matmul(y.T,x)
    output=np.matmul(output,np.linalg.inv(b11))
    output=np.matmul(output,alph*np.identity(x.shape[1]).T)
    output=np.matmul(output,np.linalg.inv(b11))
    return output

def b2_s2_func(x, y, alph,s2):
    mu1,cov1=weight(x, y, alph,s2)
    b12=np.matmul(x.T,x)+(s2*alph*np.identity(x.shape[1]))
    
    inside=np.matmul(s2*(-np.linalg.inv(b12)),alph*np.identity(x.shape[1]))
    inside=np.matmul(inside,np.linalg.inv(b12))+np.linalg.inv(b12)
    
    output=np.matmul(-np.linalg.inv(cov1),inside)
    output=np.matmul(output,np.linalg.inv(cov1))
    
    
    return output

def b3_s2_func(x, y, alph,s2):
    b13=np.linalg.inv(np.matmul(x.T,x)+(s2*alph*np.identity(x.shape[1])))
    
    output=np.matmul(b13,alph*np.identity(x.shape[1]))
    output=np.matmul(output,b13)
    output=np.matmul(output,np.matmul(x.T,y))
    return output

def b_wrt_s2(x, y, alph,s2,w):
    mu1,cov1=weight(x, y, alph,s2)
    b1_s2=b1_s2_func(x, y, alph,s2)
    first_term=np.matmul(b1_s2,np.linalg.inv(cov1))
    first_term=np.matmul(first_term,(w-mu1))
    
    b2_s2=b2_s2_func(x, y, alph,s2)
    second_term=np.matmul((w-mu1).T,b2_s2)
    second_term=np.matmul(second_term,(w-mu1))
    
    b3_s2=b3_s2_func(x, y, alph,s2)
    third_term=np.matmul((w-mu1).T,np.linalg.inv(cov1))
    third_term=np.matmul(third_term,b3_s2)
    
    output=first_term+second_term+third_term

    return output


def b1_alph_func(x, y, alph,s2):
    b11=np.matmul(x.T,x)+(s2*alph*np.identity(x.shape[1]).T)
    output=np.matmul(y.T,x)
    output=np.matmul(output,np.linalg.inv(b11))
    output=np.matmul(output,s2*np.identity(x.shape[1]).T)
    output=np.matmul(output,np.linalg.inv(b11))
    return output

def b2_alph_func(x, y, alph,s2):
    mu1,cov1=weight(x, y, alph,s2)
    b12=np.matmul(x.T,x)+(s2*alph*np.identity(x.shape[1]))
    
    inside=np.matmul(s2*(-np.linalg.inv(b12)),s2*np.identity(x.shape[1]))
    inside=np.matmul(inside,np.linalg.inv(b12))
    
    output=np.matmul(-np.linalg.inv(cov1),inside)
    output=np.matmul(output,np.linalg.inv(cov1))
    
    
    return output

def b3_alph_func(x, y, alph,s2):
    b13=np.linalg.inv(np.matmul(x.T,x)+(s2*alph*np.identity(x.shape[1])))
    
    output=np.matmul(b13,s2*np.identity(x.shape[1]))
    output=np.matmul(output,b13)
    output=np.matmul(output,np.matmul(x.T,y))
    return output

def b_wrt_alph(x, y, alph,s2,w):
    mu1,cov1=weight(x, y, alph,s2)
    b1_alph=b1_alph_func(x, y, alph,s2)
    first_term=np.matmul(b1_alph,np.linalg.inv(cov1))
    first_term=np.matmul(first_term,(w-mu1))
    
    b2_alph=b2_alph_func(x, y, alph,s2)
    second_term=np.matmul((w-mu1).T,b2_alph)
    second_term=np.matmul(second_term,(w-mu1))
    
    b3_alph=b3_alph_func(x, y, alph,s2)
    third_term=np.matmul((w-mu1).T,np.linalg.inv(cov1))
    third_term=np.matmul(third_term,b3_alph)
    
    output=first_term+second_term+third_term

    return output

def b_wrt_w(x, y, alph,s2,w):
    mu1,cov1=weight(x, y, alph,s2)
    first_term=np.matmul(np.linalg.inv(cov1),(w-mu1))
    
    third_term=np.matmul((w-mu1).T,np.linalg.inv(cov1))
    return first_term+third_term
######################################
######################################
######################################

def e(x0,f):
    x=f[0]
    y=f[1]
    s2=x0[0]
    alph=x0[1]
    w=x0[2:]
    energy=0
   
    mu1,cov1=weight(x, y, alph,s2)
    b=np.matmul((w-mu1)[:,np.newaxis].T,np.linalg.inv(cov1))
    b=np.matmul(b,(w-mu1))
    energy+=b
    return energy/2
   
def g(x0,f):
    x=f[0]
    y=f[1]
    s2=x0[0]
    alph=x0[1]
    w=x0[2:]  
    #output_s2=b_wrt_s2(x, y, alph,s2)[0]
    
    output_s2=b_wrt_s2(x, y, alph,s2,w)/2
    output_alph=b_wrt_alph(x, y, alph,s2,w)/2
    output_w=(b_wrt_w(x, y, alph,s2,w)/2).tolist()
    return np.array([output_s2,output_alph]+output_w)


######################################
######################################
######################################
#alph=10
x0 = np.random.normal(size=11)
##x0=np.array([5.0,5.0])
print(x0)
f=[x_train,y_train]
hmc.gradient_check(x0, energy_function, grad_function, f)
######################################
######################################
######################################


def cal_log_prob(s2_list,alph_list):
    #alph_list,s2_list=np.meshgrid(alph_list,s2_list)
    log_p_list=[]
    i=0
    for alph in alph_list:
        row=[]
        print(i)
        for s2 in s2_list:
            energy=energy_function(np.array([s2,alph]),[x_train,y_train])
            log_p=-energy
            row.append(log_p)
        i+=1
        log_p_list.append(np.array(row))
    
    log_p_list=np.array(log_p_list)
    
    return log_p_list
alph_list=np.linspace(-0.85,-0.6,99)
s2_list=np.linspace(1.6,1.85,100)
log_p_list=cal_log_prob(s2_list,alph_list)
plt.figure()
plt.contourf(s2_list,alph_list,log_p_list)
plt.xlabel("s2")
plt.ylabel("alph")
######################################
######################################
######################################

np.random.seed(seed=1)  # For reproducibility 
#eps=0.000025
eps=0.00005
np.random.seed(seed=1)  
R = 100
burn = int(R/10)  
L = 100  
x0 = np.random.normal(size=11)


S, reject = hmc.sample(x0, energy_function, grad_function, R, L, eps, burn=burn, checkgrad=True, args=[f])
'''
plt.figure()
plt.title("log posterior")
plt.contourf(s2_list,alph_list,log_p_list)
plt.scatter(S[:, 0], S[:, 1],c="red",s=10)
plt.xlabel("s2")
plt.ylabel("alph")
'''
######################################
######################################
######################################
s2_list=S[:, 0]
alph_list=S[:, 1]
log_p_list=cal_log_prob(s2_list,alph_list)

plt.figure()
plt.title("log posterior")
plt.contourf(s2_list,alph_list,log_p_list)
plt.scatter(s2_list, alph_list,c="red",s=10)
plt.xlabel("s2")
plt.ylabel("alph")
######################################
######################################
######################################
#plot to see converge of value
num_list=np.arange(1,s2_list.shape[0]+1)
s2_list=np.cumsum(s2_list)/num_list
alph_list=np.cumsum(alph_list)/num_list

plt.figure()
plt.title("converge of s2 value")
plt.plot(num_list,s2_list)

plt.figure()
plt.title("converge of alpha value")
plt.plot(num_list,alph_list)
######################################
######################################
######################################
w,_=weight(x_train, y_train,np.mean(S[:,1]),np.mean(S[:,0]))
#w,_=weight(x_train, y_train,S[-1,1],S[-1,0])
#w,_=weight(x_train, y_train,x0[1],x0[0])
y_pred_train=np.matmul(x_train,w)
RMSE_train=np.sqrt(np.mean((y_pred_train-y_train)**2))

y_pred_test=np.matmul(x_test,w)
RMSE_test=np.sqrt(np.mean((y_pred_test-y_test)**2))
print("RMSE_train : ",RMSE_train)
print("RMSE_test : ",RMSE_test)