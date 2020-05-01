# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:59:35 2020

@author: Tisana
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

####################################################################
####################################################################
####################################################################
#task 1
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

######################################
######################################
######################################
#exploratory on each variable
plt.figure()
plt.suptitle("correlation between each attribute x and target value y")
for i in range(0,x_train.shape[1]):
    plt.subplot(2,4,i+1)
    title=attribute_x_list[i]
    plt.title(title)
    
    x=x_train[:,i]
    y=y_train
    
    corelation=np.corrcoef(x, y)
    
    plt.scatter(x,y,label=corelation)
    plt.legend()
plt.show()
######################################
######################################
######################################
#perform linear regression as baseline predictor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

#learn
regressor.fit(x_train,y_train)

#see train ans test result
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)

#compute error
train_error=np.sqrt(np.mean((y_train_pred-y_train)**2))
test_error=np.sqrt(np.mean((y_test_pred-y_test)**2))
print("train error : ",train_error)
print("test error : ",test_error)

#plot
plt.figure()
plt.suptitle("correlation between actual y and pedicted y")
plt.subplot(1,2,1)
plt.title("train")
plt.scatter(y_train,y_train,label="actual",c="red")
plt.plot(y_train_pred,y_train_pred,label="predict",c="blue")
plt.legend()
plt.subplot(1,2,2)
plt.title("test")
plt.scatter(y_test,y_test,label="actual",c="red")
plt.plot(y_test_pred,y_test_pred,label="predict",c="blue")
plt.legend()
######################################
######################################
######################################
#perform backward elimination
import statsmodels.api as sm
x=np.append(np.ones((x_train.shape[0],1)).astype(int),x_train,axis=1)

x_optimal=x[:,[0,1,2,3,4,5,6,7,8]]
regressor=sm.OLS(endog=y_train,exog=x_optimal).fit()
regressor.summary()

#remove first variable
x_optimal=x[:,[0,1,2,3,4,5,7,8]]
regressor=sm.OLS(endog=y_train,exog=x_optimal).fit()
regressor.summary()

#remove second variable
x_optimal=x[:,[0,1,2,3,4,5,7]]
regressor=sm.OLS(endog=y_train,exog=x_optimal).fit()
regressor.summary()   

#perform regression with optimal x
regressor=LinearRegression()

#learn
regressor.fit(x_optimal,y_train)

#see train ans test result
x=np.append(np.ones((x_test.shape[0],1)).astype(int),x_test,axis=1)
x_optimal_test=x[:,[0,1,2,3,4,5,7]]
y_train_pred=regressor.predict(x_optimal)
y_test_pred=regressor.predict(x_optimal_test)

#compute error
train_error=np.sqrt(np.mean((y_train_pred-y_train)**2))
test_error=np.sqrt(np.mean((y_test_pred-y_test)**2))
print("train error : ",train_error)
print("test error : ",test_error)
####################################################################
####################################################################
####################################################################
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

from scipy import stats
def alpha(x, y, alph, s2):
    #### **** YOUR CODE HERE **** ####
    #a=np.matmul(x,np.transpose(x))
    #a=np.matmul(np.transpose(x),x)
   # a=a/alph
    #covariance=(s2*np.identity(a.shape[0]))+a
    cov_alph=s2*np.identity(x_train.shape[0])+np.matmul((alph**-1)*x_train,x_train.T)

    log_p=stats.multivariate_normal.logpdf(y,cov=cov_alph,allow_singular=True) 
    #log_p=sum(stats.multivariate_normal.logpdf(x,cov=covariance,allow_singular=True)) 
    return log_p


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
    
    #log_p=sum(stats.multivariate_normal.logpdf(mean,mean=mean,cov=var,allow_singular=True))
    
    log_p=(stats.multivariate_normal.logpdf(mean,mean,var,allow_singular=True))
    return log_p,mean,var

alph=5
s2=5
log_pp=weight(x_train, y_train, alph,s2)[0]+alpha(x_train, y_train, alph, s2)
def predict(mean_weight,var_weight,x_test):
    mean_pred=np.squeeze(np.matmul(mean_weight[np.newaxis,:],x_test.T))
    var_pred=np.matmul(x_test,np.matmul(var_weight,x_test.T))
    return mean_pred,var_pred
######################################
######################################
######################################
#type 2 maximum likelihood
    
best_log_alpha=0
best_log_s2=0
log_alpha_list=np.linspace(-25,10,100)
log_s2_list=np.linspace(-25,10,100)
log_list=[]
lop_p_alph_list=[]
max_value=-999999
i=0
for log_s2 in log_s2_list:
    i+=1
    #print(max_value)
    print(i)
    row_list=[]
    r=[]
    row_matrix_list=[]
    for log_alpha in log_alpha_list:
        alph=np.exp(log_alpha)
        s2=np.exp(log_s2)   
        
        log_p_alph=alpha(x_train, y_train, alph, s2)
        log_p_w,_,_=weight(x_train, y_train, alph,s2)
        lgp=log_p_alph+log_p_w
        #print(lgp)
        row_list.append(lgp)
        r.append(log_p_alph)
        row_matrix_list.append([log_alpha,log_s2])
        if lgp>max_value:
            max_value=lgp
            best_log_alpha=log_alpha
            best_log_s2=log_s2
    log_list.append(np.array(row_list))
    lop_p_alph_list.append(np.array(r))
log_list=np.array(log_list)
lop_p_alph_list=np.array(lop_p_alph_list)


print("max value ",max_value) 
print("best log alpha ",best_log_alpha)
print("best log s2 ",best_log_s2) 
plt.figure()
plt.title("log p")
plt.contourf(log_alpha_list,log_s2_list,log_list) 
plt.scatter(best_log_alpha,best_log_s2,c="red")
plt.xlabel("log alpha")
plt.ylabel("log s2")

plt.figure()
plt.title(" p")
plt.contourf(log_alpha_list,log_s2_list,np.exp(log_list)) 
plt.scatter(best_log_alpha,best_log_s2,c="red")
plt.xlabel("log alpha")
plt.ylabel("log s2")


#train ans test error
_,mean_weight,var_weight=weight(x_train, y_train, np.exp(best_log_alpha),np.exp(best_log_s2))
y_pred,_=predict(mean_weight,var_weight,x_train)
train_error=np.sqrt(np.mean((y_pred-y_train)**2))
print("RMSE train : ",train_error)

y_pred,_=predict(mean_weight,var_weight,x_test)
test_error=np.sqrt(np.mean((y_pred-y_test)**2))
print("RMSE test : ",test_error)
######################################
######################################
######################################
#variational inference
def update_q_alpha(a_0,M,b_0,m_n,s_n):
    a_n=a_0+(M/2)
    value=np.matmul(m_n.T,m_n)+np.trace(s_n)
    b_n=b_0+0.5*value
    return a_n,b_n[0][0]

def update_q_beta(c_0,M,d_0,m_n,s_n):
    c_n=c_0+(M/2)
    value=np.matmul(m_n.T,m_n)+np.trace(s_n)
    d_n=d_0+0.5*value
    return c_n,d_n[0][0]


def update_q_w(beta,x,y,alpha):
    s_n=np.linalg.inv(((alpha)*np.identity(x.shape[0]))+(beta*np.matmul(x,x.T)))
    #m_n=beta*np.matmul(s_n,np.matmul(y[np.newaxis,:],x.T).T)
    m_n=beta*np.matmul(s_n,np.matmul(y[np.newaxis,:],x.T).T)
    return m_n,s_n
######################################
######################################
######################################
a_0=1
b_0=1
c_0=1
d_0=1
M=x_train.shape[1]

x_train=x_train.T
x_test=x_test.T
######################################
######################################
######################################
a_n=a_0
b_n=b_0
c_n=c_0
d_n=d_0

l=[]
rmse__test_list=[]
rmse_train_list=[]
alpha_list=[]
beta_list=[]
for n in range(1,x_train.shape[1]*3):
    x=x_train[:,:n]
    y=y_train[:n]
    #update q w
    alpha=(a_n/b_n)
    beta=c_n/d_n
    m_n,s_n=update_q_w(beta,x,y,alpha)
    #w=stats.multivariate_normal.rvs(mean=np.squeeze(m_n),cov=s_n)
    w=np.squeeze(m_n)
    q_w=stats.multivariate_normal.pdf(w,mean=np.squeeze(m_n),cov=s_n,allow_singular=True)
    
    #update q alpha
    a_n,b_n=update_q_alpha(a_0,M,b_0,m_n,s_n)
    alpha=a_n/b_n
    alpha_list.append(alpha)
    q_alpha=(stats.gamma.pdf(x=alpha,a=a_n,scale=1/b_n))

    #update q beta

    c_n,d_n=update_q_beta(c_0,M,d_0,m_n,s_n)
    beta=c_n/d_n
    beta_list.append(beta)
    q_beta=(stats.gamma.pdf(x=beta,a=c_n,scale=1/d_n))


    
    #total prob
    log_p=q_w*q_alpha*q_beta
    #print(log_p,w)
    l.append(log_p)
    
    #pred
    y_pred_test=sum(w[:,np.newaxis]*x_test)
    rmse_test=np.sqrt(np.mean((y_test-y_pred_test)**2))
    rmse__test_list.append(rmse_test)
    
    y_pred_train=sum(w[:,np.newaxis]*x_train)
    rmse_train=np.sqrt(np.mean((y_train-y_pred_train)**2))
    rmse_train_list.append(rmse_train)
    print(alpha,beta)

plt.figure()
plt.title("posterior")
plt.plot(np.arange(0,len(l)),l)  
plt.xlabel("number of iteration")
plt.ylabel("posterior")  

plt.figure()
plt.title("train and test error")
plt.plot(np.arange(0,len(rmse_train_list)),rmse_train_list,label="train error")
plt.plot(np.arange(0,len(rmse__test_list)),rmse__test_list,label="train error")
plt.xlabel("number of iteration")
plt.legend()
plt.ylabel("RMSE")  

print("RMSE train : ",rmse_train_list[-1])
print("RMSE test : ",rmse__test_list[-1])
best_alpha=alpha
best_beta=beta
print("best  alpha ",best_alpha)
print("best  beta ",best_beta) 


#plot
beta_list=np.array(beta_list)
alpha_list=np.array(alpha_list)
alpha_list,beta_list=np.meshgrid(alpha_list,beta_list)
weight_list=[]
for i in range(0,alpha_list.shape[0]):
    print(i)
    row_list=[]
    for j in range(0,alpha_list.shape[1]):
        alpha=alpha_list[i,j]
        beta=beta_list[i,j]
        weight=update_q_w(beta,x,y,alpha)[0]
        row_list.append(weight)
    weight_list.append(np.array(row_list))
weight_list=np.array(weight_list)
weight_list=weight_list[:,:,:,0]


q_alpha_list=(stats.gamma.pdf(x=alpha_list,a=a_n,scale=1/b_n))
q_beta_list=(stats.gamma.pdf(x=beta_list,a=c_n,scale=1/d_n))
q_w_list=stats.multivariate_normal.pdf(weight_list,mean=np.squeeze(m_n),cov=s_n,allow_singular=True)


total_prob=q_w_list*q_alpha_list*q_beta_list
total_log_prob=q_w_list+q_alpha_list+q_beta_list

plt.figure()
plt.contourf(alpha_list,beta_list,total_log_prob)
plt.xlabel(" alpha")
plt.ylabel(" beta (1/s2)")
plt.scatter(best_alpha,best_beta,c="red")
####################################################################
####################################################################
####################################################################