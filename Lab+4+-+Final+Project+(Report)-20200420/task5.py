# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:19:48 2020

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

#transform the output
y_train_binary=np.zeros(y_train.shape[0])
y_test_binary=np.zeros(y_test.shape[0])

y_train_binary[y_train>23]=1
y_train_binary[y_train<=23]=0

y_test_binary[y_test>23]=1
y_test_binary[y_test<=23]=0

#parameter
a0=10**(-2)
b0=10**(-4)
####################################################################
####################################################################
####################################################################

def sigmoid_func(w,x):
    z=np.matmul(x,w)
    output=1/(1+np.exp(-z))
    return output
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
    p=sigmoid_func(w,x)
    a=sum((y*np.log(p))+((1-y)*np.log(1-p)))
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
    
    return energy
    

 
####################################################################
####################################################################
####################################################################
#cehck energy function
'''
x0 = np.random.normal(size=11)
x0[0]=np.exp(x0[0])
x0[1]=np.exp(x0[1])
f=[x_train,y_train_binary]
x=f[0]
y=f[1]
s2=x0[0]
alph=x0[1]
w=x0[2:]

p=sigmoid_func(w,x)
a_real=stats.bernoulli.logpmf(y,p).sum()
a=energy_function(x0,f)
print("check a : ", np.isclose(a,a_real))
'''
####################################################################
####################################################################
####################################################################
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

    
    #a wrt w
    p=sigmoid_func(w,x)
    a_w=np.matmul((y-p),x)
    output_w=output_w-a_w

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

####################################################################
####################################################################
####################################################################

#check gradient
x0 = np.random.normal(size=11)
x0[0]=np.exp(x0[0])
x0[1]=np.exp(x0[1])
##x0=np.array([5.0,5.0])
print(x0)
f=[x_train,y_train_binary]
hmc.gradient_check(x0, energy_function, grad_function, f)

####################################################################
####################################################################
####################################################################


np.random.seed(seed=1)  # For reproducibility 
eps=0.00038
R = 10000
burn = int(R/10)  
L = 100
x0 = np.random.normal(size=11)
x0[0]=np.exp(x0[0])
x0[1]=np.exp(x0[1])

S, reject = hmc.sample(x0, energy_function, grad_function, R, L, eps, burn=burn, checkgrad=True, args=[f])

####################################################################
####################################################################
####################################################################
#plot to see convergence of estimate avlue
s2_list=S[:,0]
alph_list=S[:,1]
w_list=S[:,2:]
num=np.arange(1,s2_list.shape[0]+1)

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
####################################################################
####################################################################
####################################################################
def accuracy_func(actual,pred):
    count_correct=(actual==pred)*1
    accuracy=sum(count_correct)/count_correct.shape[0]
    return accuracy
    
    
#see accuracy
final_alph=alph_list[-1]
final_s2=s2_list[-1]
final_w=w_list.sum(axis=0)/num[-1]

print("best s2 : ",final_s2)
print("best alph : ",final_alph)
print("best w : ",final_w)
threshold=0.5

y_pred_train=sigmoid_func(final_w,x_train)
y_pred_train[y_pred_train>threshold]=1
y_pred_train[y_pred_train<=threshold]=0

y_pred_test=sigmoid_func(final_w,x_test)
y_pred_test[y_pred_test>threshold]=1
y_pred_test[y_pred_test<=threshold]=0

train_accuracy=accuracy_func(y_pred_train,y_train_binary)
test_accuracy=accuracy_func(y_pred_test,y_test_binary)

print("accuracy train : ",train_accuracy)
print("accuracy test : ",test_accuracy)

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
####################################################################
####################################################################
####################################################################
'''
def s(w,x,e):
    z=np.matmul(x,w)+e
    output=1/(1+np.exp(-z))
    return output

def e(x0,f):
    x=f[0]
    y=f[1]
    s2=x0[0]
    alph=x0[1]
    w=x0[2:-1]
    ee=x0[-1]
    energy=0
    term=(b0**a0)/gamma(a0)
    N=x.shape[0]
    M=x.shape[1]  
    #a
    p=s(w,x,ee)
    a=sum((y*np.log(p))+((1-y)*np.log(1-p)))
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
    
    #noise term
    noise=(-0.5*np.log(2*np.pi))-(ee**2)/2
    energy-=noise
    
    return energy
def g(x0,f):
    x=f[0]
    y=f[1]
    s2=x0[0]
    alph=x0[1]
    w=x0[2:-1]    
    ee=x0[-1]
   #common term
    N=x.shape[0]
    M=x.shape[1]    
    output_w=np.array([0]*len(w))
    output_s2=0
    output_alph=0  
    output_e=0
    
    #a wrt w
    p=s(w,x,ee)
    a_w=np.matmul((y-p),x)
    output_w=output_w-a_w  
    
    #a wrt ee
    a_e=sum(y-p)
    output_e=output_e-a_e      

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
    
    #noise wrt ee
    noise_e=-ee
    output_e=output_e-noise_e
    
    
    return np.array([output_s2,output_alph]+output_w.tolist()+[output_e])

x0 = np.random.normal(size=12)
x0[0]=np.exp(x0[0])
x0[1]=np.exp(x0[1])
##x0=np.array([5.0,5.0])
print(x0)
f=[x_train,y_train_binary]
hmc.gradient_check(x0, e, g, f)

####################################################################
####################################################################
####################################################################

np.random.seed(seed=1)  # For reproducibility 
eps=0.00028
R = 10000
burn = int(R/10)  
L = 100
x0 = np.random.normal(size=11)
x0[0]=np.exp(x0[0])
x0[1]=np.exp(x0[1])

S, reject = hmc.sample(x0, e, g, R, L, eps, burn=burn, checkgrad=True, args=[f])


####################################################################
####################################################################
####################################################################
#plot to see convergence of estimate avlue
s2_list=S[:,0]
alph_list=S[:,1]
w_list=S[:,2:-1]
e_list=S[:,-1]
num=np.arange(1,s2_list.shape[0]+1)

s2_list=np.cumsum(s2_list)/num
alph_list=np.cumsum(alph_list)/num
w1_list=np.cumsum(w_list[:,0])/num
e_list=np.cumsum(e_list)/num

plt.figure()
plt.title("convergence of s2")
plt.plot(num,s2_list)

plt.figure()
plt.title("convergence of alph")
plt.plot(num,alph_list)

plt.figure()
plt.title("convergence of w1")
plt.plot(num,w1_list)

plt.figure()
plt.title("convergence of e")
plt.plot(num,e_list)
    
#see accuracy
final_alph=alph_list[-1]
final_s2=s2_list[-1]
final_w=w_list.sum(axis=0)/num[-1]
final_e=e_list[-1]

print("best s2 : ",final_s2)
print("best alph : ",final_alph)
print("best w : ",final_w)
print("best e : ",final_e)

threshold=0.5

y_pred_train=s(final_w,x_train,final_e)
y_pred_train[y_pred_train>threshold]=1
y_pred_train[y_pred_train<=threshold]=0

y_pred_test=s(final_w,x_test,final_e)
y_pred_test[y_pred_test>threshold]=1
y_pred_test[y_pred_test<=threshold]=0

train_accuracy=accuracy_func(y_pred_train,y_train_binary)
test_accuracy=accuracy_func(y_pred_test,y_test_binary)

print("accuracy train : ",train_accuracy)
print("accuracy test : ",test_accuracy)
'''
