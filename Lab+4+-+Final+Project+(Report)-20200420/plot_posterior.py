# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:45:43 2020

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
def energy_function(x0,ff):
    x=ff[0]
    y=ff[1]
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
def log_posterior(x0,f):
    log_p=-energy_function(x0,f)
    return log_p

######################################
######################################
######################################
a=0.006737946999085467
b=0.13462209254849022
c=4.5399929762484854e-05
d=9.271688191719882
e=-7.23462662
f=-6.80564271
g=-4.54897081
h=-3.68826174
i=0.75594473
j=1.25374272
k=-4.23192596
l=-3.00750759
m=7.2039515
n=7.37815691
o=-0.12623419
p=-0.12444385
q=2.76752209
r=2.77138242
s=0.20247917
t=0.20993861
u=22.91265232
v=22.92070311

alph_list=np.linspace(a,b,99)
s2_list=np.linspace(c,d,100)
w1_list=np.linspace(e,f,100)
w2_list=np.linspace(g,h,100)
w3_list=np.linspace(i,j,100)
w4_list=np.linspace(k,l,100)
w5_list=np.linspace(m,n,100)
w6_list=np.linspace(o,p,100)
w7_list=np.linspace(q,r,100)
w8_list=np.linspace(s,t,100)
w9_list=np.linspace(u,v,100)

ff=[x_train,y_train]
######################################
######################################
######################################
#plot alph and s2


posterior_list=[]
p_alph=[]

for alph in alph_list:
    
    p_s2=[]
    for s2 in s2_list:
        x0=[s2,alph,w1_list[-1],w2_list[-1],w3_list[-1]\
            ,w4_list[-1],w5_list[-1],w6_list[-1],w7_list[-1]\
            ,w8_list[-1],w9_list[-1]]
        print(x0)
        x0=np.array(x0)
        log_prob=-energy_function(x0,[x_train,y_train])
        p_s2.append(log_prob)
    p_alph.append(np.array(p_s2))

posterior_list=np.array(p_alph)

plt.figure()
plt.title("contour btw s2 and alph ")
plt.contourf(s2_list,alph_list,posterior_list)



######################################
######################################
######################################
#compare log posterior
#HMC | VI | type 2 ML

s2=[0.016737224846942945,0.13462209254849022,0.006737946999085467]
alph=[9.271688191719882,9.260917427758185,4.5399929762484854e-05]
w1=[-6.80564271,-6.86459553,-7.23462662]
w2=[-4.54897081,-3.68826174,-3.94213963]
w3=[1.25374272,0.81494196,0.75594473]
w4=[-3.00750759,-4.01007459,-4.23192596]
w5=[7.37815691,7.31383394,7.2039515]
w6=[-0.12444385,-0.12623419,-0.12516927]
w7=[2.76752209,2.77138242,2.77021895]
w8=[0.20993861,0.20247917,0.20406264]
w9=[22.91402406,22.91265232,22.92070311]

plt.figure()
model_list=["HMC","VI","type 2 ML"]
plt.title("log posterior of each model")
for i in range(0,3):
    
    x0=[s2[i],alph[i],w1[i],w2[i],w3[i]\
                ,w4[i],w5[i],w6[i],w7[i]\
                ,w8[i],w9[i]]
    x0=np.array(x0)
    log_p=log_posterior(x0,[x_train,y_train])  
    plt.bar(i,log_p,label=model_list[i])
plt.ylabel("log posterior")
plt.legend()
plt.show()
######################################
######################################
######################################
#comapre train and test error
#HMC | VI | type 2 ML
train_error=[3.0118549466697635,3.011767383050821,3.0116]
test_error=[3.0887831894624247,3.0918273471103173,3.0959]
plt.figure()
plt.title("train error for each model")
for i in range(0,3):
    

    
    plt.bar(i,train_error[i],label=model_list[i])
    plt.ylim(ymin=3.0115,ymax=3.01191)
plt.ylabel("RMSE ")
plt.legend()
plt.show()

plt.figure()
plt.title("test error for each model")
for i in range(0,3):
    

    
    plt.bar(i,test_error[i],label=model_list[i])
    plt.ylim(ymin=3.08513,ymax=3.11603)
plt.ylabel("RMSE ")
plt.legend()
plt.show()

gap=abs(np.array(test_error)-np.array(train_error))
plt.figure()
plt.title("gap between train and test error for each model")
for i in range(0,3):
    

    
    plt.bar(i,gap[i],label=model_list[i])
    plt.ylim(ymin=0.0732,ymax=0.0863)
plt.ylabel("RMSE ")
plt.legend()
plt.show()
######################################
######################################
######################################
'''
posterior_list=[]

p_alph=[]
for alph in alph_list:
  
    p_s2=[]
    for s2 in s2_list:
        p_w1=[]
        for w1 in w1_list:
            p_w2=[]
            for w2 in w2_list:
                p_w3=[]
                for w3 in w3_list:
                    p_w4=[]
                    for w4 in w4_list:
                        p_w5=[]
                        for w5 in w5_list:
                            p_w6=[]
                            for w6 in w6_list:
                                p_w7=[]
                                for w7 in w7_list:
                                    p_w8=[]
                                    for w8 in w8_list:
                                        p_w9=[]
                                        for w9 in w9_list:
                                            x0=[s2,alph,w1,w2,w3,w4\
                                                ,w5,w6,w7,w8,w9]
                                            x0=np.array(x0)
                                            prob=posterior(x0,[x_train,y_train])
                                            p_w9.append(prob)
                                        p_w8.append(np.array(p_w9)) 
                                    p_w7.append(np.array(p_w8))
                                p_w6.append(np.array(p_w7))
                            p_w5.append(np.array(p_w6))
                        p_w4.append(np.array(p_w5))
                    p_w3.append(np.array(p_w4))
                p_w2.append(np.array(p_w3))
            p_w1.append(np.array(p_w2))
        p_s2.append(np.array(p_w1))
    p_alph.append(np.array(p_s2))

posterior_list.append(np.array(p_alph))
'''

######################################
######################################
######################################
case_list=["type 2 ML","over fit","under fit"]
train_error=[3.0116,3.0828,5.9431]
test_error=[3.0959,3.0977,4.81124]
gap=abs(np.array(test_error)-np.array(train_error))

plt.figure()
plt.title("train error")
plt.ylabel("RMSE")
for i in range(0,3):
    

    
    plt.bar(i,train_error[i],label=case_list[i])
    #plt.ylim(ymin=0.0732,ymax=0.0863)
plt.ylabel("RMSE ")
plt.legend()
plt.show()

plt.figure()
plt.title("test error")
plt.ylabel("RMSE")
for i in range(0,3):
    

    
    plt.bar(i,test_error[i],label=case_list[i])
    #plt.ylim(ymin=0.0732,ymax=0.0863)
plt.ylabel("RMSE ")
plt.legend()
plt.show()