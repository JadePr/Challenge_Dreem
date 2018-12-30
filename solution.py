#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:12:51 2018

@author: jade
"""
#Import packages

import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import cmath
from scipy.fftpack import rfft, irfft, fftfreq
from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

#Import dataset
train = pd.read_csv("/home/jade/Documents/Biotech/Machine Learning/Challenge/train.csv", nrows = 50000)


#Remove outliers : Remove eeg signals that go beyond a threshold value



eeg=train.filter(regex='eeg_*',axis=1)
vmx=np.max(eeg, axis=1)
vmi=np.min(eeg, axis=1)
df_train=train.loc[(vmx<400) & (vmi>-400)]



#Create Fourier transform of the EEG signal and return Fourier coefficients and squared Fourier coefficients


def fourier_eeg(data_train,listsqr,listphi,splitrate): 
    for i in np.arange(splitrate):
        f, t, temp =signal.stft(data_train.iloc[i], fs=250, nperseg=400)

        
        listsqr.append(abs(temp)**2)
        
        listphi.append(abs(temp))
    return listsqr,listphi,f;

fouriersqr=[]
fourier=[]
fouriersqr,fourier,f =fourier_eeg(df_train.filter(regex='eeg_*',axis=1),fouriersqr,fourier,df_train.count())

#The direct use of the breathing signal or its Fourier transform was proved irrelevant, which is why we do not use it in the following study

#Compute the sum the absolute values of eeg signal
som=np.sum(abs(df_train.filter(regex='eeg_*',axis=1).iloc[:,1250:].values), axis=1)
som=pd.DataFrame(som)
som=np.asarray(som)

#Compute the standard deviation of eeg signal
std=np.std(df_train.filter(regex='eeg_*',axis=1).iloc[:,1250:].values, axis=1)
std=pd.DataFrame(std)
std=np.asarray(std)

#Compute the mean of eeg signal
mean=np.mean(df_train.filter(regex='eeg_*',axis=1).iloc[:,1250:].values, axis=1)
mean=pd.DataFrame(mean)
mean=np.asarray(mean)

#Frequency condition for delta waves, charachteristic of deep sleep
cond2=[[(f>0.5) & (f<4)]]

#Compute energy with the sum of Fourier coefficients

wavestft=[]
wavephi=[]
for i in np.arange(df_train.count()):
    wavestft.append([fouriersqr[i][c][:,5:-1] for c in cond2 ])
    wavephi.append([fourier[i][c][:,5:-1] for c in cond2 ])

#Reshape data
wavestft=np.asarray(wavestft)
sz=1
sh=np.shape(wavestft)
for i in np.arange(len(sh)-1):
    sz=sz*sh[i+1] 
wavestft=np.reshape(wavestft,(df_train.count(), sz))
wavephi=np.asarray(wavephi)
wavephi=np.reshape(wavephi,(df_train.count(), sz))




#Concatenate features

metatrain=df_train[['time_previous','number_previous','time','user','night']]


X_train=np.concatenate((wavestft, wavephi, som, std, mean, metatrain), axis=1)
Y_train=df_train['power_increase']

# Feature Scaling
sc=StandardScaler()
X_std=sc.fit_transform(X_train)


#Implement Gradient Boosting Regression - 
params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'huber', 'subsample':0.5}
clf = ensemble.GradientBoostingRegressor(**params)
clf_fit=clf.fit(X_std, Y_train)

# Compute root mean squared error
RMSE_clf = np.sqrt(mean_squared_error(Y_test,Y_clf))
RMSE_clf
### GRID SEARCHES + PLOTS 




grid = GridSearchCV(clf,
                   {'alpha': [0.5,0.7,0.9,0.99],
                  }, verbose=1)
grid.fit(X_std, Y_train)
print(grid.best_score_)
print(grid.best_params_)


#Compute loss function
test_score=np.zeros(params['n_estimators'],dtype=np.float64)

for i, Y_clf in enumerate(clf.staged_predict(X_stdtest)):
    test_score[i]= clf.loss_(Y_test, Y_clf)

#Plot results
plt.figure()

plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(300 ), clf_reference.train_score_, 'g-',
         label='Training Set Deviance (ref)')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.plot(np.arange(300 ), test_score_ref, '-',
         label='Test Set Deviance (ref)')
plt.legend(loc='upper right')



#Save the model

filename = 'regressor50000.sav'
joblib.dump(clf, 'regressor50000.sav')

#Load model
loaded_model = joblib.load('regressor.sav')
Y_up = loaded_model.predict(X_stdtest)


#As a comparison, other regression techniques have been tried such as Lasso, Ridge, Kernel Regression models, Support Vector Machines. Because the results were less performant, we did not display these trials in this study.




#Plot according to features importance

feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
##plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance, align='center')
plt.xlim((10e-5,100))

plt.xscale('linear')

