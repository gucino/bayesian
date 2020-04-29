# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:11:06 2020

@author: Tisana
"""
'''
import numpy as np
import matplotlib.pyplot as plt


slope=50
x=np.random.normal(10,1,1000)
y=slope*x




x_transpose=np.transpose(x)
estimated_slope=np.matmul(x_transpose,x)
#inverse=np.linalg.inv(estimated_slope)
inverse=1/estimated_slope
answer=inverse*x_transpose
answer=np.matmul(answer,y)
'''


import numpy as np
import matplotlib.pyplot as plt


p=0.3
x=np.random.binomial(1,p,1000000)

slope=0.1 ##parameter to be estimate
y=slope*x


#assume prior is beta (alpha=1,beta=1)
alpha=1
beta=1


num_observation=100000
post_alpha=alpha+sum(y[:num_observation])
post_beta=beta+num_observation-sum(y[:num_observation])
post_data=np.random.beta(post_alpha,post_beta,10000)
plt.hist(post_data)
post_mean=post_alpha/(post_alpha+post_beta)
