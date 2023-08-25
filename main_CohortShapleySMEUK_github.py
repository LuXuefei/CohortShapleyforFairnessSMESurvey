# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:33:44 2023

@author: xuefei.lu, xuefei.lu@skema.edu
"""

import os
import urllib
import math
import numpy as np
import pandas as pd
from scipy.stats import distributions
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from cohortshapley import similarity
from cohortshapley import cohortshapley as cs


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
path = 'C:\\Users\\xuefei.lu\\...'
os.chdir(path)
#%% Dataset preparation
filename = 'Sought_Got_data_notcovid.xlsx'
df = pd.read_excel(filename,index_col=False)
df = df[df['Got_finance'].notna()]
# drop irrelavent columns
df = df.loc[:, ~df.columns.isin(['uniq_id','serial_number', 'wave','Sought_finance'])]

# Drop instances with unclear answers and prepare data for calculation
df = df.drop(df[df.ethnicity_group == 'Not answered'].index)
df = df.drop(df[df.ethnicity_group == 'Refused / DK'].index)  
df = df.drop(df[df.turnover == "Refused"].index)  
df = df.drop(df[df.turnover == "Don't know"].index)

#%% Digitalise and Define subgroups
# Based on European Commission standards \url{https://single-market-economy.ec.europa.eu/smes/sme-definition_en}

region = {"London":1, "East of Engand":2, 'South West':3 ,'Wales':4 ,
             'South East':5, 'West Midlands':6, 'East Midlands':7,   'Yorkshire/Humberside':8, 
          'North West':9, 'North/North East':10, 'Scotland':11, 'Northern Ireland':12}
df = df.replace({"region": region})

risk = {"1 - Minimal":1, "2 - Low":2, '3 - Average':3 ,'4 - Above Average':4 , '5 - Not known':5}
df = df.replace({"risk": risk})

employment_size = {"1":1, "2-10":1, '11-50':2 ,'51-100':3 , '101-200':3, '201-250':3}
employment_size_name = {"micro":1, 'small':2 , 'medium':3}
df = df.replace({"employment_size": employment_size})

turnover = { 'Less than 25,000 (12.5k)':1 ,'25,000 - 49,999 (37.5k)':1 ,
             '50,000 - 74,999 (62.5k)':1, '75,000 - 99,999 (87.5k)':1, '100,000 - 249,999 (175k)':1,  
             '250,000 - 499,999 (375k)':1, 
          '500,000 - 999,999 (750k)':1, '1m - 1.9m (1.5m)':1,  #micro
          '2m-4.9m (3.5m)':2, '5m - 9.9m (7.5m)':2, #small
          '10m - 14.9m (12.5m)':3, '15m-24.9m (20m)':3 }#median
turnover_name = { 'micro':1 ,
          'small':2, #small
          'median':3 } #median
df = df.replace({"turnover": turnover})

industry = {
    'Real Estate, Renting and Business Activities': 1,
    'Construction': 2,
    'Other Community, Social and Personal Service Activities': 3,
    'Transport, Storage and Communication': 4,
    'Wholesale / Retail': 5,
    'Agriculture, Hunting and Forestry, Fishing': 6,
    'Manufacturing': 7,
    'Hotels and Restaurants': 8,
    'Health and Social Work': 9
}
df = df.replace({"industry": industry})

legal= {
    'Sole Proprietor': 1,
    'Partnership': 2,
    'LLP&LLC': 3}
df = df.replace({"legal": legal})

age = {'Less than 12 months ago': 1,
    'Over 1 but under 2 years ago': 2,
    '2 - 5 years ago': 2,
    '6 - 9 years ago': 2,
    '10 - 15 years ago': 2,
    'More than 15 years ago': 2}
age_name = {'start-up': 1,
    'non start-up': 2}
df = df.replace({"age": age})

women_lead = {'No': 0, 'Yes': 1}
df = df.replace({"women_lead": women_lead})
df['women_lead'].value_counts(dropna = False) 
women_lead_name = {'women_lead':2, 'men_lead':1}


ethnicity_group = {
    'White - British': 1,
    'White - Irish': 1,
    'Any other white background': 1,
    'Black or Black British - African': 2,
    'Black or Black British - Caribbean': 2,
    'Black or Black British - Any other Black background': 2,
    'Mixed - White and Black African': 3,
    'Mixed - White and Black Caribbean': 3,
    'Mixed - White and Asian': 3,
    'Mixed - Any other mixed background': 3,
    'Asian or Asian British - Indian': 4,
    'Asian or Asian British - Pakistani': 4,
    'Asian or Asian British - Bangladeshi': 4,
    'Asian or Asian British - Any other Asian background': 4,
    'Chinese or ethnic group - Other ethnic group': 4,
    'Chinese or ethnic group - Chinese':4
}
ethnicity_group_name={'White':1,  'Black or Black British':2, 'Mixed': 3, 'Asian or Asian British':4}

df = df.replace({"ethnicity_group": ethnicity_group})
df['ethnicity_group'].value_counts(dropna = False) 

Got_finance = {'No': 0, 'Yes': 1}
df = df.replace({"Got_finance": Got_finance})
df['Got_finance'].value_counts(dropna = False) 

#%% prepare data for calculation

Xweight = df['weight']
X = df.drop(columns = ['weight','Got_finance'])
Y = df['Got_finance']

# # For Cross Validation
# from sklearn.model_selection import train_test_split
# df_train, df_test, dfy_train, dfy_test = train_test_split(df,  df['Got_finance'], test_size=0.2, random_state=42)

# Xweight = df_train['weight']
# X = df_train.drop(columns = ['weight','Got_finance'])
# Y = dfy_train.copy()

#%% Build Artifical bias systems to calculate RFS
subject = X.values
similarity.bins = 20
f=False #no Machine leanring model

########################################################
# True system -- weighted avg cohort shapley
cs_obj_weight = cs.CohortShapley(f, similarity.similar_in_samebin, np.arange(len(subject)), subject, 
                            y=Y.values, parallel=4,data_weight=Xweight)

cs_obj_weight.compute_cohort_shapley()
cs_obj_weight.shapley_values

########################################################
# Gender weighted - bias min

# Bias system - Gender
Ybias_gender_min = np.array([0 if w > 0 else 1 for w in X['women_lead']])
#{0: 437, 1: 665}

cs_obj_weight_gendermin = cs.CohortShapley(f, similarity.similar_in_samebin, np.arange(len(subject)), subject, 
                            y=Ybias_gender_min, parallel=4,data_weight=Xweight)

cs_obj_weight_gendermin.compute_cohort_shapley()
cs_obj_weight_gendermin.shapley_values

########################################################
# ethnicity weighted - bias min
# Bias system - ethnicity
Ybias_ethnicity_min= np.array([1 if w < 3 else 0 for w in X['ethnicity_group']])
# {0: 59, 1: 1043}

cs_obj_weight_ethnicitymin = cs.CohortShapley(f, similarity.similar_in_samebin, np.arange(len(subject)), subject, 
                            y=Ybias_ethnicity_min, parallel=4,data_weight=Xweight)

cs_obj_weight_ethnicitymin.compute_cohort_shapley()
cs_obj_weight_ethnicitymin.shapley_values

########################################################
# employment_size - bias min

# Bias system - employment_size
Ybias_employment_size_min = np.array([0 if w < 2 else 1 for w in X['employment_size']])

cs_obj_weight_employmentsizemin = cs.CohortShapley(f, similarity.similar_in_samebin, np.arange(len(subject)), subject, 
                            y=Ybias_employment_size_min, parallel=4,data_weight=Xweight)

cs_obj_weight_employmentsizemin.compute_cohort_shapley()
cs_obj_weight_employmentsizemin.shapley_values

########################################################
# age - bias min

# Bias system - age
Ybias_age_min = np.array([0 if w < 2 else 1 for w in X['age']])

cs_obj_weight_agemin = cs.CohortShapley(f, similarity.similar_in_samebin, np.arange(len(subject)), subject, 
                            y=Ybias_age_min, parallel=4,data_weight=Xweight)

cs_obj_weight_agemin.compute_cohort_shapley()
cs_obj_weight_agemin.shapley_values


########################################################
# turnover - bias min

# Bias system - turnover
Ybias_turnover_min = np.array([0 if w < 2 else 1 for w in X['turnover']])


cs_obj_weight_turnovermin = cs.CohortShapley(f, similarity.similar_in_samebin, np.arange(len(subject)), subject, 
                            y=Ybias_turnover_min, parallel=4,data_weight=Xweight)

cs_obj_weight_turnovermin.compute_cohort_shapley()
cs_obj_weight_turnovermin.shapley_values

checkgroup = Ybias_turnover_min
unique, counts = np.unique(checkgroup, return_counts=True)
dict(zip(unique, counts))

#%% define functions for calculation and plotting
### binning
def binningmy(X, bins=20):
    bin_indices = []
    bin_info_bins = []
    bin_info_x = []
    for j in range(X.shape[1]):
        n_bins = bins
        v = X[:,j]
        bin_vals = np.unique(v)
        rep_x = None
        #print(bin_vals)
        if len(bin_vals) < n_bins:
            bin_vals = np.sort(bin_vals)
            n_bins = len(bin_vals)
            rep_x = np.array(bin_vals)
        else:
            bin_vals = np.linspace(v.min(),v.max(), n_bins+1, endpoint=True)
            rep_x = np.zeros(n_bins)
            for i in range(n_bins):
                rep_x[i] = (bin_vals[i] + bin_vals[i+1])/2
        bin_index = np.digitize(v, bin_vals, right= False)
        bin_indices.append(bin_index)
        bin_info_bins.append(bin_vals)
        bin_info_x.append(rep_x)
    return np.array(bin_indices).T, bin_info_bins, bin_info_x

bin_X = binningmy(X.values)
bin_idx = bin_X[0]
bin_val = bin_X[1]

# plot functions
def conditioned_shapley_hist(X, bin_val, shapley_values, condvar, expvar, bin_label=None, weights = None, meanline = None, ylim=(0,2000), xlim=None):
    colororder =plt.get_cmap("tab10")#plt.rcParams['axes.prop_cycle'].by_key()['color']
    vals = bin_val[condvar]
    col = X.columns
    n_vals = len(vals)
    WAscore = []
    cond = {}
    for k in range(n_vals):
        cond[k] = np.where(X[col[condvar]] == vals[k])[0]
    jj = expvar
    for k in range(n_vals):
        v = shapley_values[:,jj]
        n_bins = 25 # nr of bins in the hist
        bins = np.linspace(v.min(),v.max()+0.001, n_bins+1, endpoint=True)
        if bin_label:
            l = list(bin_label.keys())[list(bin_label.values()).index(k+1)]#bin_label[k]
        else:
            l = k
            
        if weights is not None:
            plt.hist(shapley_values[cond[k]][:,jj], bins=bins, weights = weights[cond[k]], alpha=0.5,label=l,color=colororder(k))
        else:
            plt.hist(shapley_values[cond[k]][:,jj], bins=bins, alpha=0.5,label=l,color=colororder(k))  
        if meanline:
            plt.axvline(x=np.mean(shapley_values[cond[k]][:,jj]),ymax=1, color=colororder(k)) #colororder[k]
        if bin_label:    
            #print(list(bin_label.keys())[list(bin_label.values()).index(k+1)]+' avg = '+str(np.mean(shapley_values[cond[k]][:,jj])))
            print(list(bin_label.keys())[list(bin_label.values()).index(k+1)]+' weighted avg = '+str(np.average(shapley_values[cond[k]][:,jj],weights = Xweight.values[cond[k]])))
        else:
            #print(str(k)+' avg = '+str(np.mean(shapley_values[cond[k]][:,jj])))
            print(str(k)+' weighted avg = '+str(np.average(shapley_values[cond[k]][:,jj],weights = Xweight.values[cond[k]])))
        
        WAscore.append(np.average(shapley_values[cond[k]][:,jj],weights = Xweight.values[cond[k]]))

    plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    
    return np.array(WAscore)   
        

def ks_weighted(data1, data2, wei1, wei2, alternative='two-sided'):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
    cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
    d = np.max(np.abs(cdf1we - cdf2we))
    # calculate p-value
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = m * n / (m + n)
    if alternative == 'two-sided':
        prob = distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the larger of (n1, n2)
        expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
        prob = np.exp(expt)
    return d, prob

#%% Results for feature: gender
condvar = 7 # conditional var - diff colors
expvar = 7 # explain var - weights
X.columns[condvar]
df[X.columns[condvar]].value_counts(dropna = False) 

# weighted
plt.figure(figsize=(11,7))
plt.rcParams["font.size"] = 15
WAscore_g = conditioned_shapley_hist(X, bin_val, cs_obj_weight.shapley_values,
                         condvar,expvar, weights=Xweight.values, meanline = True, ylim = None,
                        bin_label=women_lead_name)
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title('Gender')
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('women_lead_cs.pdf', bbox_inches='tight')  
plt.show()


# bias sys - min
plt.figure(figsize=(10,7))
plt.rcParams["font.size"] = 15
WAscore_g_min = conditioned_shapley_hist(X, bin_val, cs_obj_weight_gendermin.shapley_values,
                         condvar,expvar, weights=Xweight.values, meanline = True, ylim = None,
                         bin_label=women_lead_name)
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title('Gender, bias min')
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('women_lead_csmin.pdf', bbox_inches='tight')  
plt.show()

# KS test
vals = bin_val[condvar]
n_vals = len(vals)
cond = {}
col = X.columns
for k in range(n_vals):
    cond[k] = np.where(X[col[condvar]] == vals[k])[0]

i1 = 0
i2 = 1
shapley_values = cs_obj_weight.shapley_values 
data1 = shapley_values[cond[i1]][:,expvar]
data2 = shapley_values[cond[i2]][:,expvar]
wei1 = Xweight.values[cond[i1]]
wei2 = Xweight.values[cond[i2]]
d, prob = ks_weighted(data1, data2, wei1, wei2)
print(d,prob)

#RFS
WAscore_g_max = -WAscore_g_min
women_lead_rscore = pd.DataFrame(data = np.vstack([WAscore_g, WAscore_g_min, WAscore_g_max]), 
                                 columns= list(women_lead.keys()))

rscore = women_lead_rscore
rscoretemp = [(rscore.iloc[0,i] - min(rscore.iloc[:,i]) )/ (max(rscore.iloc[:,i]) - min(rscore.iloc[:,i])) for i in range(len(rscore.columns)) ]
rscore.loc[len(rscore.index)] = rscoretemp
rscore.index = ['org','biasmin','biasmax','RFS']
print(rscore)

# store RFS
RFS = []
RFS_name = []

RFS.extend(rscore.loc['RFS'].tolist())
RFS_name.extend([f'women_lead_{col}' for col in rscore.columns])

#%% Results for feature: ethnicity_group
condvar = 8 
expvar = 8 
X.columns[condvar]
df[X.columns[condvar]].value_counts(dropna = False) 

plt.figure(figsize=(10,7))
plt.rcParams["font.size"] = 15
WAscore_e = conditioned_shapley_hist(X, bin_val, cs_obj_weight.shapley_values,condvar,expvar,
                         bin_label=ethnicity_group_name,weights=Xweight.values,
                         meanline = True,ylim=(0,150))
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title(X.columns[condvar])
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('ethnicity_group_cs.pdf', bbox_inches='tight')  
plt.show()

# bias system - min
plt.figure(figsize=(10,7))
plt.rcParams["font.size"] = 15
WAscore_e_min = conditioned_shapley_hist(X, bin_val, cs_obj_weight_ethnicitymin.shapley_values,condvar,expvar,
                         bin_label=ethnicity_group_name,weights=Xweight.values,
                         meanline = True,ylim=(0,150))
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title(X.columns[condvar] + ', bias min')
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('ethnicity_group_csmin.pdf', bbox_inches='tight')  
plt.show()

# RFS
WAscore_e_max = -WAscore_e_min
ethnicity_group_rscore = pd.DataFrame(data = np.vstack([WAscore_e, WAscore_e_min, WAscore_e_max]), 
                                 columns= list(ethnicity_group_name.keys()))

rscore = ethnicity_group_rscore
rscoretemp = [(rscore.iloc[0,i] - min(rscore.iloc[:,i]) )/ (max(rscore.iloc[:,i]) - min(rscore.iloc[:,i])) for i in range(len(rscore.columns)) ]
rscore.loc[len(rscore.index)] = rscoretemp
rscore.index = ['org','biasmin','biasmax','RFS']
print(rscore)

RFS.extend(rscore.loc['RFS'].tolist())
RFS_name.extend([f'eth_grp_{col}' for col in rscore.columns])

#KS test
vals = bin_val[condvar]
n_vals = len(vals)
cond = {}
col = X.columns
for k in range(n_vals):
    cond[k] = np.where(X[col[condvar]] == vals[k])[0]

# change group index here
i1 = 2
i2 = 3
shapley_values = cs_obj_weight.shapley_values #cs_obj_weight_gendermin.shapley_values
data1 = shapley_values[cond[i1]][:,expvar]
data2 = shapley_values[cond[i2]][:,expvar]
wei1 = Xweight.values[cond[i1]]
wei2 = Xweight.values[cond[i2]]
d, prob = ks_weighted(data1, data2, wei1, wei2)
print(d,prob)


#%% Results for feature: employment_size
condvar = 2 
expvar = 2 
X.columns[condvar]
df[X.columns[condvar]].value_counts(dropna = False) 

plt.figure(figsize=(10,7))
plt.rcParams["font.size"] = 15
WAscore_ep = conditioned_shapley_hist(X, bin_val, cs_obj_weight.shapley_values,condvar,expvar,
                         bin_label=employment_size_name,weights=Xweight.values, 
                         meanline = True,ylim=None)
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title(X.columns[condvar])
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('employment_size_cs.pdf', bbox_inches='tight')  
plt.show()


# bias system
plt.figure(figsize=(10,7))
plt.rcParams["font.size"] = 15
WAscore_ep_min= conditioned_shapley_hist(X, bin_val, cs_obj_weight_employmentsizemin.shapley_values,condvar,expvar,
                         bin_label=employment_size_name,weights=Xweight.values, 
                         meanline = True,ylim=None)
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title(X.columns[condvar]+ ', bias min')
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('employment_size_csmin.pdf', bbox_inches='tight')  
plt.show()


# RFS
WAscore_ep_max = -WAscore_ep_min
employment_size_rscore = pd.DataFrame(data = np.vstack([WAscore_ep, WAscore_ep_min, WAscore_ep_max]), 
                                 columns= list(employment_size_name.keys()))

rscore = employment_size_rscore
rscoretemp = [(rscore.iloc[0,i] - min(rscore.iloc[:,i]) )/ (max(rscore.iloc[:,i]) - min(rscore.iloc[:,i])) for i in range(len(rscore.columns)) ]
rscore.loc[len(rscore.index)] = rscoretemp
rscore.index = ['org','biasmin','biasmax','RFS']
print(rscore)

RFS.extend(rscore.loc['RFS'].tolist())
RFS_name.extend([f'emp_size_{col}' for col in rscore.columns])


# Weighted KS test
vals = bin_val[condvar]
n_vals = len(vals)
cond = {}
col = X.columns
for k in range(n_vals):
    cond[k] = np.where(X[col[condvar]] == vals[k])[0]

# change group index here
i1 = 1
i2 = 2
shapley_values = cs_obj_weight.shapley_values #cs_obj_weight_gendermin.shapley_values
data1 = shapley_values[cond[i1]][:,expvar]
data2 = shapley_values[cond[i2]][:,expvar]
wei1 = Xweight.values[cond[i1]]
wei2 = Xweight.values[cond[i2]]
d, prob = ks_weighted(data1, data2, wei1, wei2)
print(d,prob)

#%% Results for feature: age
condvar = 6 # conditional var - diff colors
expvar = 6 # explain var - weights
X.columns[condvar]
print(df['age'].value_counts(dropna = False) )


plt.figure(figsize=(10,7))
plt.rcParams["font.size"] = 15
WAscore_age = conditioned_shapley_hist(X, bin_val, cs_obj_weight.shapley_values,condvar,expvar,
                         bin_label=age_name,weights=Xweight.values, 
                         meanline = True,ylim=None)
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title(X.columns[condvar])
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('age_cs.pdf', bbox_inches='tight')  
plt.show()


#bias min
plt.figure(figsize=(10,7))
plt.rcParams["font.size"] = 15
WAscore_age_min = conditioned_shapley_hist(X, bin_val, cs_obj_weight_agemin.shapley_values,condvar,expvar,
                         bin_label=age_name,weights=Xweight.values, 
                         meanline = True,ylim=None)
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title(X.columns[condvar] + ', bias min')
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('age_csmin.pdf', bbox_inches='tight')  
plt.show()


# RFS
WAscore_age_max = -WAscore_age_min
age_rscore = pd.DataFrame(data = np.vstack([WAscore_age, WAscore_age_min, WAscore_age_max]), 
                                 columns= list(age_name.keys()))

rscore = age_rscore
rscoretemp = [(rscore.iloc[0,i] - min(rscore.iloc[:,i]) )/ (max(rscore.iloc[:,i]) - min(rscore.iloc[:,i])) for i in range(len(rscore.columns)) ]
rscore.loc[len(rscore.index)] = rscoretemp
rscore.index = ['org','biasmin','biasmax','RFS']
print(rscore)

RFS.extend(rscore.loc['RFS'].tolist())
RFS_name.extend([f'age_{col}' for col in rscore.columns])


# KS test
vals = bin_val[condvar]
n_vals = len(vals)
cond = {}
col = X.columns
for k in range(n_vals):
    cond[k] = np.where(X[col[condvar]] == vals[k])[0]

# change group index here
i1 = 0
i2 = 1
shapley_values = cs_obj_weight.shapley_values #cs_obj_weight_gendermin.shapley_values
data1 = shapley_values[cond[i1]][:,expvar]
data2 = shapley_values[cond[i2]][:,expvar]
wei1 = Xweight.values[cond[i1]]
wei2 = Xweight.values[cond[i2]]
d, prob = ks_weighted(data1, data2, wei1, wei2)
print(d,prob)
#%% Results for feature: turnover
condvar = 3 # conditional var - diff colors
expvar = 3 # explain var - weights
X.columns[condvar]
print(df[X.columns[condvar]].value_counts(dropna = False) )

plt.figure(figsize=(10,7))
plt.rcParams["font.size"] = 15
WAscore_turnover = conditioned_shapley_hist(X, bin_val, cs_obj_weight.shapley_values,condvar,expvar,
                         bin_label=turnover_name,weights=Xweight.values, 
                         meanline = True,ylim=None)
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title(X.columns[condvar])
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('turnover_cs.pdf', bbox_inches='tight')  
plt.show()

plt.figure(figsize=(10,7))
plt.rcParams["font.size"] = 15
WAscore_turnover_min = conditioned_shapley_hist(X, bin_val, cs_obj_weight_turnovermin.shapley_values,condvar,expvar,
                         bin_label=turnover_name,weights=Xweight.values, 
                         meanline = True,ylim=None)
plt.legend(bbox_to_anchor=(1,0), loc='lower left')
plt.title(X.columns[condvar] + ', bias min')
plt.xlabel('Impact')
plt.ylabel('Frequency')
plt.grid(visible=True,axis='y')
#plt.savefig('turnover_csmin.pdf', bbox_inches='tight')  
plt.show()

# RFS
WAscore_turnover_max = -WAscore_turnover_min
turnover_rscore = pd.DataFrame(data = np.vstack([WAscore_turnover, WAscore_turnover_min, WAscore_turnover_max]), 
                                 columns=list(turnover_name.keys()))

rscore = turnover_rscore
rscoretemp = [(rscore.iloc[0,i] - min(rscore.iloc[:,i]) )/ (max(rscore.iloc[:,i]) - min(rscore.iloc[:,i])) for i in range(len(rscore.columns)) ]
rscore.loc[len(rscore.index)] = rscoretemp
rscore.index = ['org','biasmin','biasmax','RFS']
print(rscore)

RFS.extend(rscore.loc['RFS'].tolist())
RFS_name.extend([f'turnover_{col}' for col in rscore.columns])



# KS test
vals = bin_val[condvar]
n_vals = len(vals)
cond = {}
col = X.columns
for k in range(n_vals):
    cond[k] = np.where(X[col[condvar]] == vals[k])[0]

# change group index here
i1 = 1
i2 = 2
shapley_values = cs_obj_weight.shapley_values #cs_obj_weight_gendermin.shapley_values
data1 = shapley_values[cond[i1]][:,expvar]
data2 = shapley_values[cond[i2]][:,expvar]
wei1 = Xweight.values[cond[i1]]
wei2 = Xweight.values[cond[i2]]
d, prob = ks_weighted(data1, data2, wei1, wei2)
print(d,prob)
