# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:05:45 2020

@author: Tisana
"""
import numpy as np
import matplotlib.pyplot as plt


parameter=0.5




data_set=np.random.binomial(1,parameter, size=100000)
alpha=100
beta=1

def haha(num_observation,sigma_x):
    posterior_alpha=alpha+sigma_x
    posterior_beta=beta+num_observation-sigma_x
    
    beta_data=np.random.beta(posterior_alpha,posterior_beta,10000)
    
    return beta_data,posterior_alpha,posterior_beta


for num_observation in range(0,1000,10):
    sigma_x=sum(data_set[:num_observation])
    beta_data,posterior_alpha,posterior_beta=haha(num_observation,sigma_x)
    plt.hist(beta_data)
    print("predicted value : ",posterior_alpha/(posterior_alpha+posterior_beta))

