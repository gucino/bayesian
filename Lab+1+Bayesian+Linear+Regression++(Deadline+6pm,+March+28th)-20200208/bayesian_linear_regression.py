# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:12:03 2020

@author: Tisana
"""

# Standard modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# Lab-specific support module
import cm50268_lab1_setup as lab1
#
N_train = 12
N_val   = N_train
N_test  = 250
#
sigma = 0.1
s2    = sigma**2


# Data - create generator instance, and synthesise 3 sets
#
generator = lab1.DataGenerator(noise=sigma)
#
(x_train, t_train) = generator.get_data('TRAIN', N_train)
(x_val, t_val) = generator.get_data('VALIDATION', N_val)
(x_test, t_test) = generator.get_data('TEST', N_test)

# Basis - create generator instance and compute the basis matrices for all 3 data sets
# Note that because we use a "bias" function, we need N-1 Gaussians to make the
# basis "complete" (i.e. for M=N)
#
M = N_train-1
r = 1 # Basis radius or width
centres = np.linspace(generator.xmin, generator.xmax, M)
basis = lab1.RBFGenerator(centres, width=r, bias=True)
#
PHI_train = basis.evaluate(x_train)
PHI_val = basis.evaluate(x_val)
PHI_test = basis.evaluate(x_test)

############################################################################
############################################################################
############################################################################
#task1a

## FIT_PLS
##
def fit_pls(PHI, t, lam):
    #
    #### **** YOUR CODE HERE **** ####
    #
    x=PHI
    y=t
    
    transpose_x=np.transpose(x)
    
    a=np.matmul(transpose_x,x)
    b=lam*np.identity(a.shape[0])
    c=np.matmul(transpose_x,y)
    
    w=np.matmul(np.linalg.inv(a+b),c)
    w=np.squeeze(w)
    return w
############################################################################
############################################################################
############################################################################
#task1b
def plot_regression(x_train,y_train,x_test,y_test,y_pred,title,variance=0):
 
    plt.figure(title)
    plt.title(title)
    plt.scatter(x_train,y_train,label="train data",c="black")
    plt.plot(x_test,y_test,label="test data",c="blue")
    #plt.plot(x_test,y_pred,label="predicted data",c="red")
    plt.errorbar(x_test, y_pred, variance,label="predicted data",c="red")
    plt.ylim([-1.5,1.5])
    plt.legend()
    plt.show()
 
for lam in [0,0.01,10]:
    title="task 1b lambda="+str(lam)
    weight=fit_pls(PHI_train, t_train, lam)
    y_pred=(weight*PHI_test).sum(axis=1)
    plot_regression(x_train,t_train,x_test,t_test,y_pred,title)


############################################################################
############################################################################
############################################################################
#task2a
## POSTERIOR
##
def compute_posterior(PHI, t, alph, s2):
    #### **** YOUR CODE HERE **** ####
    x=PHI
    y=t
    lam=alph*s2
    
    transpose_x=np.transpose(x)
    
    a=np.matmul(transpose_x,x)
    b=lam*np.identity(a.shape[0])
    c=np.matmul(transpose_x,y)
    
    Mu=np.matmul(np.linalg.inv(a+b),c)
    Mu=np.squeeze(Mu)
    
    SIGMA=s2*(np.linalg.inv(a+b))
    return Mu, SIGMA
lam=0.01
weight=fit_pls(PHI_train, t_train, lam)
alph=lam/s2
mu,sigma=compute_posterior(PHI_train, t_train, alph, s2)
print("check : ",np.isclose(mu,weight).all())

############################################################################
############################################################################
############################################################################
#task2b
## MARGINAL LIKELIHOOD
##
def compute_log_marginal(PHI, t, alph, s2):
    #### **** YOUR CODE HERE **** ####
    a=np.matmul(PHI,np.transpose(PHI))
    a=a/alph
    covariance=(s2*np.identity(a.shape[0]))+a
    
    lgp=stats.multivariate_normal.logpdf(np.squeeze(t),cov=covariance) 
    return lgp
a=compute_log_marginal(PHI_train,t_train,alph,s2)

############################################################################
############################################################################
############################################################################
#task2c
def error_rms(y_pred,y_test):
    rmse=np.sqrt(np.mean((y_pred-y_test)**2))
    return rmse

log_lambda_value_to_try=np.linspace(-5,5,100)


train_rmse_list=[]
test_rmse_list=[]
val_rmse_list=[]
negative_log_likelihood_list=[]
alpha_list=[]
for log_lambda in log_lambda_value_to_try:
    lam=10**log_lambda
    #OLS
    weight=fit_pls(PHI_train, t_train, lam)
    train_pred=(weight*PHI_train).sum(axis=1)
    test_pred=(weight*PHI_test).sum(axis=1)
    val_pred=(weight*PHI_val).sum(axis=1)
    
    train_rmse_list.append(error_rms(train_pred,np.squeeze(t_train)))
    test_rmse_list.append(error_rms(test_pred,np.squeeze(t_test)))
    val_rmse_list.append(error_rms(val_pred,np.squeeze(t_val)))
    
    #Bayesian
    alph=lam/s2
    alpha_list.append(alph)
    log_maginal=compute_log_marginal(PHI_train, t_train, alph, s2)
    negative_log_likelihood_list.append(log_maginal*-1)
    
plt.figure("task2c")
plt.title("task2c")
plt.ylim([0,0.7])
plt.plot(log_lambda_value_to_try,train_rmse_list,label="train error",c="black")
plt.plot(log_lambda_value_to_try,test_rmse_list,label="test error",c="yellow")
plt.plot(log_lambda_value_to_try,val_rmse_list,label="validation error",c="blue")
plt.legend()
plt.gca().twinx()
#plt.ylim([-3,5])
plt.plot(log_lambda_value_to_try,negative_log_likelihood_list,label="Bayesian ",c="red")
plt.legend()
plt.show()

#compute lowest error
#find index of lowest point
min_index_test=test_rmse_list.index(min(test_rmse_list))
min_index_val=val_rmse_list.index(min(val_rmse_list))
min_index_bayesian=negative_log_likelihood_list.index(min(negative_log_likelihood_list))

print("min test error : ",test_rmse_list[min_index_test])
print("min val error : ",test_rmse_list[min_index_val])
print("min bayesian (neg log marginal) error : ",test_rmse_list[min_index_bayesian])

############################################################################
############################################################################
############################################################################
#task2d
title="task2d"
alph=(10**log_lambda_value_to_try[min_index_bayesian])/s2
print("best alpha for bayesian : ",alph)
mu,sigma=compute_posterior(PHI_train, t_train, alph, s2)
y_pred=(mu*PHI_test).sum(axis=1)

plot_regression(x_train,t_train,x_test,t_test,y_pred,title)

############################################################################
############################################################################
############################################################################
#task3a
mu,sigma=compute_posterior(PHI_train, t_train, alph, s2)
variance=np.matmul((PHI_test),sigma)
variance=(variance*PHI_test).sum(axis=1)
sd=np.sqrt(variance)

title="task3a"
y_pred=(mu*PHI_test).sum(axis=1)
plot_regression(x_train,t_train,x_test,t_test,y_pred,title,sd)

############################################################################
############################################################################
############################################################################
#task3b
