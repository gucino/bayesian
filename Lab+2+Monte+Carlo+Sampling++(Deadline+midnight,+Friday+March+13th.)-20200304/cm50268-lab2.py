# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:27:11 2020

@author: Tisana
"""

##
## Setup
##
# Standard modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# Lab-specific support module
import cm50268_lab2_setup as lab2

##
## Setup
##
## Define some fixed values
##

# Parameters that determine the generated data
#
sig_gen = 0.2  # Standard deviation
s2_gen = sig_gen**2  # Variance
r_gen = 1  # Basis function width used to generate the data
x_max = 10  # x-limit of the data
#
N_train = 30
N_test = 250

# Parameters that determine the basis set used for modelling
# - note that the length scale "r" will be varied
#
M = 16 # Number of functions, spaced equally
centres = np.linspace(0, x_max, M)

# Generate training data
seed = 4
Data = lab2.DataGenerator(m=9, r=r_gen, noise=sig_gen, rand_offset=seed)
x_train, t_train = Data.get_data('TRAIN', N_train)
x_test, t_test = Data.get_data('TEST', N_test)

# Demonstrate use of basis
r = r_gen * 0.5  # Example model uses basis functions that are too narrow
RBF = lab2.RBFGenerator(centres, r) # centres was fixed earlier
PHI_train = RBF.evaluate(x_train)
PHI_test = RBF.evaluate(x_test)

# Find posterior mean for fixed guesses for alpha and s2
alph = 1e-12
s2 = 0.1**2
mu, _ = lab2.compute_posterior(PHI_train, t_train, alph, s2)
y_test = PHI_test @ mu

# Show the training data and generating function, plus our mean fit
lab2.plot_regression(x_train, t_train, x_test, t_test, y_test)
plt.title("Data, Underlying Function & Example Predictor")
pass







##################################################################
##################################################################
##################################################################
##task1a

s2=0.2**2
def log_prob_alph_r_given_t(alph, r, s2, x, t, centres):
    #recreate RBF object
    RBF = lab2.RBFGenerator(centres, r) 
    
    #recalculate basis matrix
    PHI = RBF.evaluate(x)
    
    #calculate logmarginal likelihood
    lgp=lab2.compute_log_marginal(PHI, t, alph, s2)
    return lgp
a=log_prob_alph_r_given_t(alph, r, s2, x_train, t_train, centres)
##################################################################
##################################################################
##################################################################
##task1b  
best_log_alpha=0
best_log_r=0
log_alpha_list=np.linspace(-9,6,100)
log_r_list=np.linspace(-2,2,100)
log_list=[]
max_value=-999999
i=0
for log_r in log_r_list:
    i+=1
    print(i)
    row_list=[]
    row_matrix_list=[]
    for log_alpha in log_alpha_list:
        alph=10**log_alpha
        r=10**log_r   
        lgp=log_prob_alph_r_given_t(alph, r, s2, x_train, t_train, centres)
        row_list.append(lgp)
        row_matrix_list.append([log_r,log_alpha])
        if lgp>max_value:
            max_value=lgp
            best_log_alpha=log_alpha
            best_log_r=log_r
    log_list.append(np.array(row_list))
log_list=np.array(log_list)

plt.figure("task1b (log 𝑝(𝛼,𝑟|𝐭))")
plt.title("task1b (log 𝑝(𝛼,𝑟|𝐭))")
plt.contourf(log_alpha_list,log_r_list,log_list)

plt.figure("task1b (𝑝(𝛼,𝑟|𝐭))")
plt.title("task1b (𝑝(𝛼,𝑟|𝐭))")
plt.contourf(log_alpha_list,log_r_list,np.exp(log_list)) 

print("max value ",max_value) 
print("best log alpha ",best_log_alpha)
print("best log r ",best_log_r)       
##################################################################
##################################################################
##################################################################
##task2a
def importance(num_samples, pstar, qstar, qrvs, fun):
    value_list=[]
    weight_list=[]
    expectation_list=[]
    for each_sample in range(0,num_samples):
        #sample from proposal distribution
        x=qrvs()
        
        #compute weight
        weight=pstar(x)/qstar(x)
        weight_list.append(weight)
        
        #compute numnerator
        value_list.append(fun(x))
        numerator=sum(np.array(value_list)*np.array(weight_list))
        #compute expectation
        expectation=numerator/sum(weight_list)
        
        expectation_list.append(expectation)
        
    return expectation_list
##################################################################
##################################################################
##################################################################
##task2b
def fun(x):
    value=10**x[1]
    return value
def pstar(x):
    log_alph=x[0]
    log_r=x[1] 
    alph=10**log_alph
    r=10**log_r   
    log_marginal_likelihood=log_prob_alph_r_given_t(alph,r,s2,x_train,t_train,centres)
    marginal_likelihood=np.exp(log_marginal_likelihood)
    return marginal_likelihood

#proposal ditrbution 1
def qrvs1():
    from scipy.stats import uniform
    log_alpha_list=np.linspace(-9,6,100)
    log_r_list=np.linspace(-2,2,100)
    '''
    logalph=np.random.choice(log_alpha_list)
    logr=np.random.choice(log_r_list)
    '''
    #compute distribution
    log_alpha_unif_dist=uniform(log_alpha_list[0],log_alpha_list[-1]-log_alpha_list[0])
    log_r_unif_dist=uniform(log_r_list[0],log_r_list[-1]-log_r_list[0])
    #sampling
    logalph=log_alpha_unif_dist.rvs()
    logr=log_r_unif_dist.rvs()
    return [logalph,logr]
def qstar1(x):
    from scipy.stats import uniform
    log_alpha_unif_dist=uniform(log_alpha_list[0],log_alpha_list[-1]-log_alpha_list[0])
    log_r_unif_dist=uniform(log_r_list[0],log_r_list[-1]-log_r_list[0])
    
    log_pdf_alpha=log_alpha_unif_dist.logpdf(x[0])
    log_pdf_r=log_r_unif_dist.logpdf(x[1])
    log_qstar=log_pdf_alpha+log_pdf_r
    qstar=np.exp(log_qstar)
    return qstar

#proposal ditrbution 2
def qrvs2():
    from scipy.stats import multivariate_normal

    #sampling
    logalph=multivariate_normal.rvs(mean=-1.5,cov=15/6)
    logr=multivariate_normal.rvs(mean=0,cov=4/6)  
    return [logalph,logr]

def qstar2(x):
    from scipy.stats import multivariate_normal

    
    log_pdf_alpha=multivariate_normal.logpdf(x[0],mean=0,cov=15/6)
    log_pdf_r=multivariate_normal.logpdf(x[1],mean=0,cov=4/6)
    log_qstar=log_pdf_alpha+log_pdf_r
    qstar=np.exp(log_qstar)
    return qstar


#proposal ditrbution 2
def qrvs3():
    from scipy.stats import multivariate_normal

    #sampling
    logalph=multivariate_normal.rvs(mean=best_log_alpha,cov=15/8)
    logr=multivariate_normal.rvs(mean=best_log_r,cov=4/8)  
    return [logalph,logr]

def qstar3(x):
    from scipy.stats import multivariate_normal

    
    log_pdf_alpha=multivariate_normal.logpdf(x[0],mean=best_log_alpha,cov=15/8)
    log_pdf_r=multivariate_normal.logpdf(x[1],mean=best_log_r,cov=4/8)
    log_qstar=log_pdf_alpha+log_pdf_r
    qstar=np.exp(log_qstar)
    return qstar
num_samples=1000
neglect=5 #neglect first 5 samples
expectation=importance(num_samples, pstar, qstar1, qrvs1, fun)
plt.figure("task2b (proposal dist=uniform)")
plt.title("task2b (proposal dist=uniform)")
plt.plot(np.arange(num_samples)[neglect:],expectation[neglect:])
print("<r> : ",expectation[-1])

expectation=importance(num_samples, pstar, qstar2 ,qrvs2, fun)
plt.figure("task2b (proposal dist=gaussian(mean center))")
plt.title("task2b (proposal dist=gaussian(mean center))")
plt.plot(np.arange(num_samples)[neglect:],expectation[neglect:])
print("<r> : ",expectation[-1])

expectation=importance(num_samples, pstar, qstar3 ,qrvs3, fun)
plt.figure("task2b (proposal dist=gaussian(mean max))")
plt.title("task2b (proposal dist=gaussian(mean max))")
plt.plot(np.arange(num_samples)[neglect:],expectation[neglect:])
print("<r> : ",expectation[-1])
##################################################################
##################################################################
##################################################################
##task2c(Bonus)
case1_all_expectation=[]
case2_all_expectation=[]
case3_all_expectation=[]
num_samples=1000
for i in range(0,100):
    print(i)
    expectation1=importance(num_samples, pstar, qstar1, qrvs1, fun)
    case1_all_expectation.append(expectation1)
    
    expectation2=importance(num_samples, pstar, qstar2 ,qrvs2, fun)
    case2_all_expectation.append(expectation2)
    
    expectation3=importance(num_samples, pstar, qstar3 ,qrvs3, fun)
    case3_all_expectation.append(expectation3)
#calculate variance
case1_variance=np.var(np.array(case1_all_expectation),axis=0)
case2_variance=np.var(np.array(case2_all_expectation),axis=0)
case3_variance=np.var(np.array(case3_all_expectation),axis=0)
all_case_var=[case1_variance,case2_variance,case3_variance]
#plot varincae
title="task 2c case"
plt.figure(title)
plt.title(title)
label_list=["case1","case2","case3"]
for i in range(0,len(all_case_var)):

    plt.plot(np.arange(num_samples)[100:],all_case_var[i][100:],label=label_list[i])
plt.legend()

##################################################################
##################################################################
##################################################################
##task3a
def metropolis(num_samples, pstar, qstar,qrvs, x0,gaussian_length_scale):
    current_sample=x0
    accepted_list=[]
    rejected_list=[]
    for i in range(0,num_samples):
        #draw new sample depneds on current state
        new_sample=qrvs(current_sample,gaussian_length_scale)
        
        #compute acceptance-rejection
    
        numerator=pstar(new_sample)*qstar(current_sample,new_sample,gaussian_length_scale**2)
        denominator=pstar(current_sample)*qstar(new_sample,current_sample,gaussian_length_scale**2)
        ratio=min(1,numerator/denominator)
        #make choice
        rand_num=np.random.random()
        if rand_num<=ratio:
            #accpet
            current_sample=new_sample
            accepted_list.append(current_sample)
        else:
            #reject
            current_sample=current_sample
            accepted_list.append(current_sample)
            rejected_list.append(new_sample)
    accepted_list=np.array(accepted_list)
    rejected_list=np.array(rejected_list)
    return accepted_list,rejected_list
##################################################################
##################################################################
##################################################################
##task3b
def qrvs(current_state,gaussian_length_scale):
    from scipy.stats import multivariate_normal
    current_log_alpha=current_state[0]
    current_log_r=current_state[1]
    logalph=multivariate_normal.rvs(mean=current_log_alpha,cov=gaussian_length_scale**2)
    logr=multivariate_normal.rvs(mean=current_log_r,cov=gaussian_length_scale**2)
    return [logalph,logr]

def qstar(sample1,sample2,gaussian_length_scale):
    from scipy.stats import multivariate_normal
    #generate pdf from each sample1 conditioned on sample 2
    log_pdf_alpha=multivariate_normal.logpdf(sample1[0],mean=sample2[0],cov=gaussian_length_scale**2)
    log_pdf_r=multivariate_normal.logpdf(sample1[1],mean=sample2[1],cov=gaussian_length_scale**2)
    log_qstar=log_pdf_alpha+log_pdf_r
    qstar=np.exp(log_qstar)
    return qstar

def pstar(x):
    log_alph=x[0]
    log_r=x[1] 
    alph=10**log_alph
    r=10**log_r   
    log_marginal_likelihood=log_prob_alph_r_given_t(alph,r,s2,x_train,t_train,centres)
    marginal_likelihood=np.exp(log_marginal_likelihood)
    return marginal_likelihood
def cal_acceptance_rate(num_samples,rejected):
    value=(num_samples-len(rejected)) / num_samples
    return value

def cal_expectation_list_r(samples,num_samples,neglect_burn_in):
    #take log out
    #samples=10**samples
    #neglect 5% of sample from burn in
    start_from=num_samples*neglect_burn_in
    start_from=int(start_from)
    samples=samples[start_from:,:]
    
    cumulativ_list=np.cumsum(samples[:,1])
    
    demoninator_list=np.arange(1,num_samples+1-start_from)
    expectation_list=cumulativ_list/demoninator_list
    return expectation_list

#assessing length scale
'''
s2=0.2**2
gaussian_length_scale=0.2
num_samples=1000
length_scale_list=np.linspace(0.0000001,1,500)
acceptance_ratio_list=[]
i=0
for gaussian_length_scale in length_scale_list:
    print(i)
    i+=1
    x0=[np.random.choice(log_alpha_list),np.random.choice(log_r_list)]

    samples, rejected = metropolis(num_samples, pstar, qstar,qrvs, x0,gaussian_length_scale)
    print(" ",len(rejected))
    acceptance_ratio=cal_acceptance_rate(num_samples,rejected)
    acceptance_ratio_list.append(acceptance_ratio)
    print(" ",acceptance_ratio)
    
case1_length_scale=[]
case2_length_scale=[]
case3_length_scale=[]

for each_acceptance_ratio in  acceptance_ratio_list:
    index=acceptance_ratio_list.index(each_acceptance_ratio)
    target_length_scale=length_scale_list[index]
    if each_acceptance_ratio>0.8:
        case1_length_scale.append(target_length_scale)
    elif each_acceptance_ratio<0.05:
        case2_length_scale.append(target_length_scale)
    elif each_acceptance_ratio<0.3 and each_acceptance_ratio>0.2:
        case3_length_scale.append(target_length_scale)
case1_ls=np.mean(case1_length_scale)
case2_ls=np.mean(case2_length_scale)
case3_ls=np.mean(case3_length_scale)
print("length scale for : ",)   
print(" case1 : ",case1_ls)
print(" case2 : ",case2_ls)
print(" case3 : ",case3_ls)
'''
length_scale_list=[0.10621251422845693,0.8435837348040908,0.2721292367754376]
neglect_burn_in=0.05
s2=0.2**2
num_samples=10000

for i in range(0,len(length_scale_list)):
    print("case : ",i)
    gaussian_length_scale=length_scale_list[i]
    print(" length scale used : ",gaussian_length_scale)
    x0=[np.random.choice(log_alpha_list),np.random.choice(log_r_list)]
    
    samples, rejected = metropolis(num_samples, pstar, qstar,qrvs, x0,gaussian_length_scale)
    
    acceptance_ratio=cal_acceptance_rate(num_samples,rejected)
    print(" acceptance ratio : ",acceptance_ratio)
    expectation=cal_expectation_list_r(10**samples,num_samples,neglect_burn_in)
    title="task 3b (case "+str(i+1)+")"
    plt.figure(title)
    plt.title(title)
    plt.plot(np.arange(0,num_samples-(num_samples*neglect_burn_in)),expectation)
    print(" <r> : ",expectation[-1])
    
    
    title="task3b (𝑝(𝛼,𝑟|𝐭))"+" case "+str(i+1)
    plt.figure(title)
    plt.title(title)
    plt.contourf(log_alpha_list,log_r_list,log_list)
    #overlay sample
    plt.scatter(rejected[:,0],rejected[:,1],label="rejected sample",c="red",s=1)
    plt.scatter(samples[:,0],samples[:,1],label="accepted sample",c="green",s=1)
    plt.legend()
##################################################################
##################################################################
##################################################################
##task3c

def qrvs(current_state,gaussian_length_scale):
    from scipy.stats import multivariate_normal
    current_log_alpha=current_state[0]
    current_log_r=current_state[1]
    current_log_s2=current_state[2]
    logalph=multivariate_normal.rvs(mean=current_log_alpha,cov=gaussian_length_scale**2)
    logr=multivariate_normal.rvs(mean=current_log_r,cov=gaussian_length_scale**2)
    logs2=multivariate_normal.rvs(mean=current_log_s2,cov=gaussian_length_scale**2)
    return [logalph,logr,logs2]

def qstar(sample1,sample2,gaussian_length_scale):
    from scipy.stats import multivariate_normal
    #generate pdf from each sample1 conditioned on sample 2
    log_pdf_alpha=multivariate_normal.logpdf(sample1[0],mean=sample2[0],cov=gaussian_length_scale**2)
    log_pdf_r=multivariate_normal.logpdf(sample1[1],mean=sample2[1],cov=gaussian_length_scale**2)
    log_pdf_s2=multivariate_normal.logpdf(sample1[2],mean=sample2[2],cov=gaussian_length_scale**2)

    log_qstar=log_pdf_alpha+log_pdf_r+log_pdf_s2
    qstar=np.exp(log_qstar)
    return qstar

def pstar(x):
    log_alph=x[0]
    log_r=x[1] 
    log_s2=x[2] 
    alph=10**log_alph
    r=10**log_r  
    s2=10**log_s2  
    
    log_marginal_likelihood=log_prob_alph_r_given_t(alph,r,s2,x_train,t_train,centres)
    marginal_likelihood=np.exp(log_marginal_likelihood)
    #(" pstar : ",marginal_likelihood)
    return marginal_likelihood
def cal_expectation_list_v(samples,num_samples,neglect_burn_in):
    #take log out

    #neglect 5% of sample from burn in
    start_from=num_samples*neglect_burn_in
    start_from=int(start_from)
    samples=samples[start_from:,:]
    
    #take only variance
    cumulativ_list=np.cumsum(samples[:,2])
    
    demoninator_list=np.arange(1,num_samples+1-start_from)
    expectation_list=cumulativ_list/demoninator_list
    
    #convert variance to sd
    #expectation_list=np.sqrt(expectation_list)
    return expectation_list



#assessing length scale
'''
num_samples=1000
length_scale_list=np.linspace(0.0000001,1,500)
acceptance_ratio_list=[]
i=0
for gaussian_length_scale in length_scale_list:
    print(i)
    i+=1
    x0=[np.random.choice(log_alpha_list),np.random.choice(log_r_list),np.random.choice(log_s2_list)]

    samples, rejected = metropolis(num_samples, pstar, qstar,qrvs, x0,gaussian_length_scale)

    acceptance_ratio=cal_acceptance_rate(num_samples,rejected)
    acceptance_ratio_list.append(acceptance_ratio)
    print(" ",acceptance_ratio)
    
case1_length_scale=[]
case2_length_scale=[]
case3_length_scale=[]

for each_acceptance_ratio in  acceptance_ratio_list:
    index=acceptance_ratio_list.index(each_acceptance_ratio)
    target_length_scale=length_scale_list[index]
    if each_acceptance_ratio>0.8:
        case1_length_scale.append(target_length_scale)
    elif each_acceptance_ratio<0.05:
        case2_length_scale.append(target_length_scale)
    elif each_acceptance_ratio<0.3 and each_acceptance_ratio>0.2:
        case3_length_scale.append(target_length_scale)
case1_ls=np.mean(case1_length_scale)
case2_ls=np.mean(case2_length_scale)
case3_ls=np.mean(case3_length_scale)
print("length scale for : ",)   
print(" case1 : ",case1_ls)
print(" case2 : ",case2_ls)
print(" case3 : ",case3_ls)
'''

log_s2_list=np.linspace(-2,-1,100)
gaussian_length_scale=0.36112857090117734#0.3670034303030303
neglect_burn_in=0.05
num_samples=10000
print(" length scale used : ",gaussian_length_scale)
x0=[np.random.choice(log_alpha_list),np.random.choice(log_r_list),np.random.choice(log_s2_list)]

samples, rejected = metropolis(num_samples, pstar, qstar,qrvs, x0,gaussian_length_scale)

acceptance_ratio=cal_acceptance_rate(num_samples,rejected)
print(" acceptance ratio : ",acceptance_ratio)
expectation=cal_expectation_list_v(10**samples,num_samples,neglect_burn_in)
title="task 3c "
plt.figure(title)
plt.title(title)
plt.plot(np.arange(0,num_samples-(num_samples*neglect_burn_in)),expectation)
print(" ⟨𝜎⟩ : ",np.sqrt(expectation[-1]))

##################################################################
##################################################################
##################################################################
##task3d
from scipy.stats import multivariate_normal
np.random.seed(5)

#log
log_s2_list=np.linspace(-2,-1,100)
log_alpha_list=np.linspace(-9,6,100)
log_r_list=np.linspace(-2,2,100)


def pstar(x):
    log_alph=x[0]
    log_r=x[1] 
    log_s2=x[2]
    w=x[3]
    
    alph=10**log_alph
    r=10**log_r  
    s2=10**log_s2  

    RBF = lab2.RBFGenerator(centres, r) 
    PHI_train = RBF.evaluate(x_train)
    
    N=t_train.shape[0]
    a=(-N/2)*np.log(2*np.pi*s2)
    b=(sum((np.squeeze(t_train)-((PHI_train*w).sum(axis=1)))**2))/(2*s2)
    log_likelihood=a-b
    
    M=w.shape[0]
    c=(M/2)*np.log(alph/(2*np.pi))
    d=(sum((w**2)*alph))/2
    log_marginal=c-d
    log_pdf=log_likelihood+log_marginal
    #(" pstar : ",marginal_likelihood)
    pstar=np.exp(log_pdf)
    return pstar

def qstar(sample1,sample2,gaussian_length_scale):
    '''
    from scipy.stats import multivariate_normal
    #generate pdf from each sample1 conditioned on sample 2
    log_pdf_alpha=multivariate_normal.logpdf(sample1[0],mean=sample2[0],cov=gaussian_length_scale**2)
    log_pdf_r=multivariate_normal.logpdf(sample1[1],mean=sample2[1],cov=gaussian_length_scale**2)
    log_pdf_s2=multivariate_normal.logpdf(sample1[2],mean=sample2[2],cov=gaussian_length_scale**2)
    
    log_pdf_w=0
    for i in range(0,sample1[3].shape[0]):
        log_pdf_w+=multivariate_normal.logpdf(sample1[3][i],mean=sample2[3][i],cov=gaussian_length_scale**2)
        

    log_qstar=log_pdf_alpha+log_pdf_r+log_pdf_s2+log_pdf_w
    qstar=np.exp(log_qstar)
    '''
    qstar=1
    return qstar


def qrvs(current_state,gaussian_length_scale):
    from scipy.stats import multivariate_normal
    current_log_alpha=current_state[0]
    current_log_r=current_state[1]
    current_log_s2=current_state[2]
    current_w=current_state[3]
    
    logalph=multivariate_normal.rvs(mean=current_log_alpha,cov=gaussian_length_scale**2)
    logr=multivariate_normal.rvs(mean=current_log_r,cov=gaussian_length_scale**2)
    logs2=multivariate_normal.rvs(mean=current_log_s2,cov=gaussian_length_scale**2)
    '''
    logw=[]
    for i in range(0,current_log_w.shape[0]):
        logw.append(multivariate_normal.rvs(mean=current_log_w[i],cov=gaussian_length_scale**2))
    logw=np.array(logw)
    '''
    w=multivariate_normal.rvs(mean=current_w,cov=[gaussian_length_scale**2]*len(current_w))
    '''
    w=[]
    for i in range(0,current_w.shape[0]):
        w.append(multivariate_normal.rvs(mean=current_w[i],cov=gaussian_length_scale**2))
    w=np.array(w)
    '''
    return [logalph,logr,logs2,w]

'''
num_samples=1000
length_scale_list=np.linspace(0.0000001,1,100)
acceptance_ratio_list=[]
i=0
for gaussian_length_scale in length_scale_list:
    print(i)
    i+=1
    w0=1
    w_list=[w0]
    for j in range(0,15):
        w=multivariate_normal.rvs(mean=w_list[j],cov=gaussian_length_scale)
        w_list.append(w)
    w_list=np.array(w_list)

    x0=[np.random.choice(log_alpha_list),np.random.choice(log_r_list),np.random.choice(log_s2_list),w_list]

    samples, rejected = metropolis(num_samples, pstar, qstar,qrvs, x0,gaussian_length_scale)

    acceptance_ratio=cal_acceptance_rate(num_samples,rejected)
    acceptance_ratio_list.append(acceptance_ratio)
    print(" ",acceptance_ratio)
    
case1_length_scale=[]
case2_length_scale=[]
case3_length_scale=[]

for each_acceptance_ratio in  acceptance_ratio_list:
    index=acceptance_ratio_list.index(each_acceptance_ratio)
    target_length_scale=length_scale_list[index]
    if each_acceptance_ratio>0.8:
        case1_length_scale.append(target_length_scale)
    elif each_acceptance_ratio<0.05:
        case2_length_scale.append(target_length_scale)
    elif each_acceptance_ratio<0.3 and each_acceptance_ratio>0.2:
        case3_length_scale.append(target_length_scale)
case1_ls=np.mean(case1_length_scale)
case2_ls=np.mean(case2_length_scale)
case3_ls=np.mean(case3_length_scale)
print("length scale for : ",)   
print(" case1 : ",case1_ls)
print(" case2 : ",case2_ls)
print(" case3 : ",case3_ls)
'''
np.random.seed(5)
num_samples=40000
gaussian_length_scale=0.2901745429752067 #0.30000007000000006 #0.3872054484848485 #0.3207071386363637 #0.3080808772727273
w0=1
w_list=[w0]
for i in range(0,15):
    w=multivariate_normal.rvs(mean=w_list[i],cov=gaussian_length_scale)
    w_list.append(w)
w_list=np.array(w_list)
x0=[np.random.choice(log_alpha_list),np.random.choice(log_r_list),np.random.choice(log_s2_list),w_list]

samples, rejected = metropolis(num_samples, pstar, qstar,qrvs, x0,gaussian_length_scale)
samples=samples.tolist()

#rearange the list to array with 19 dimension
new_sammple_list=[]
for i in range(0,len(samples)):
    
    ori=samples[i][:3]
    weight=list(samples[i][3])
    new_sammple_list.append(ori+weight)
new_sammple_list=np.array(new_sammple_list)
start_from=num_samples*0.10
start_from=int(start_from)
new_sammple_list=new_sammple_list[start_from:,:]
acceptance_ratio=cal_acceptance_rate(num_samples,rejected)


#compute y_pred from samples
all_y_pred=[]
#find r
neglect_burn_in=0
expectation=cal_expectation_list_r(10**new_sammple_list,len(new_sammple_list),neglect_burn_in)
variance=cal_expectation_list_v(10**new_sammple_list,len(new_sammple_list),neglect_burn_in)[-1]
print("⟨𝜎⟩  : ",np.sqrt(variance))

for i in range(0,len(new_sammple_list)):
    #print(i)
    w=new_sammple_list[i,3:]
    r=expectation[i]
    RBF = lab2.RBFGenerator(centres, r)
    PHI_test = RBF.evaluate(x_test)
    
    y_pred=PHI_test@w
    
    all_y_pred.append(y_pred)

#avg y_pred
all_y_pred=np.array(all_y_pred)
y_pred=all_y_pred.sum(axis=0)/(num_samples)
plt.figure("task3d")
lab2.plot_regression(x_train, t_train, x_test, t_test, y_pred)
plt.title("Data, Underlying Function & Example Predictor")
##################################################################
##################################################################
##################################################################













