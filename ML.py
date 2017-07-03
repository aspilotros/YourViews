# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:45:56 2017

@author: Alessandro
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
#%%
class cumsum(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,x,y=None):
        return self
    def transform(self,x):
        return x.cumsum(axis=1)
    
class SHModel(BaseEstimator, RegressorMixin):
#    train=pd.read_csv(open(train, "rb"))
#    test=pd.read_csv(open(test, "rb"))
    def __init__(self,refer_d, target_d):
        self.alpha = 0
        self.sum1 = 0
        self.sum2 = 0
        self.refer_d = refer_d
        self.target_d = target_d
        pass
    
    def fit(self, x, y = None):
        self.train_x = x.iloc[:,0:self.refer_d].values
        self.train_y = x.iloc[:,0:self.target_d].values
        self.num_of_train = x.shape[0]
        """
            calculate alpha
            S-H model: alpha * N(refer_d) = N(target_d)
        """
        for i in range(self.num_of_train):
            self.sum1 = self.sum1 +( sum(self.train_x[i]) / sum(self.train_y[i]) )
            self.sum2 = self.sum2 +( sum(self.train_x[i]) / sum(self.train_y[i]) ) ** 2        
        self.alpha = self.sum1 / self.sum2
#        print("alpha:", self.alpha)
        return self
    
    def predict (self, x):
        #self.test_x=x.iloc[:,0:self.refer_d].values
        #self.test_y=x.iloc[:,0:self.target_d].values
        self.num_of_test=x.shape[0]
        x_array=x.values
        self.predictions = []
        for i in range(self.num_of_test):
            self.predictions.append(self.alpha * sum(x_array[i]))
        prediction_final=self.predictions
        return prediction_final
    
    def mRSE_score(self,y):
        """
        calculate mRSE      
        """
        y_array=y.values
        self.RSE = []
        for i in range(self.num_of_test):            
            self.RSE.append( ( (self.predictions[i] / sum(y_array[i])) - 1) ** 2)
        self.mRSE = sum(self.RSE)/ self.num_of_test
        self.std_RSE = np.std(self.RSE)
        return self.mRSE
#%%
class MLModel(BaseEstimator, RegressorMixin):
    
    def __init__(self,refer_d, target_d):
        self.refer_d = refer_d
        self.target_d = target_d
        pass
    
    
       
#    def MLModel(train, test, refer_d, target_d):
    def fit(self, x, reg, y = None):
        
        def MLmRSE(alph, a, b):
            #return np.sqrt(np.abs(np.dot(alph, a) / b - 1))
            return np.sqrt(np.abs(np.dot(alph, a) / b - 1) + reg*np.sum(np.square(alph)))
        train_x = x.iloc[:,0:self.refer_d].values
        train_y = np.sum(x.iloc[:,0:self.target_d].values,axis=1)
        self.num_train_col = train_x.shape[1]
        self.num_train = x.shape[0]
        #test_x=test.iloc[:,0:refer_d].values
        #test_y=np.sum(test.iloc[:,0:target_d].values, axis=1)
        """
            According M-L model: alpha1*day1 + ... + alphan*dayn = N(target_d)
            get alpha by using leastsq
        """
        self.alpha = [0] * self.num_train_col
        X     = np.transpose(np.array(train_x))
        Y     = np.array(train_y)
        self.alpha = leastsq(MLmRSE, self.alpha, args = (X, Y))[0]
        return self
    def predict(self,x):        
        self.predictions = []
        self.num_test=x.shape[0]
    #    RSE = 0
        for i in range(self.num_test):
    #        RSE += ((np.dot(alpha, test_x[i]) / test_y[i]) - 1) ** 2
            self.predictions.append(np.dot(self.alpha, x.values[i]))
            
        return self.predictions
    
    def MLmRSE_score(self,y):
        y_array=np.sum(y.iloc[:,0:self.target_d].values, axis=1)
        """
            calculate mRSE
        """
        self.RSE=[]
        for i in range(self.num_test):
            self.RSE.append( ((self.predictions[i] / y_array[i])-1)**2 )
        mRSE = sum(self.RSE) / self.num_test
        self.std = np.std(self.RSE)
        return mRSE
    
    def MLmARE_score(self,y):
        y_array=np.sum(y.iloc[:,0:self.target_d].values, axis=1)
        """
            calculate average absolute relative error
        """
        self.ARE=[] #Absolute relative error
        for i in range(self.num_test):
            self.ARE.append( np.abs((self.predictions[i] / y_array[i])-1) )
        mARE = sum(self.ARE) / self.num_test
        self.std = np.std(self.ARE)
        return mARE
        