# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:18:48 2020

@author: O46743
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('.\\3. Data Pre-Processed\\v3\\SCALES_Pre-Processed_GLOBAL_training_test_v3.csv')
validation = pd.read_csv('.\\3. Data Pre-Processed\\v3\\SCALES_Pre-Processed_GLOBAL_validation_v3.csv')


# ----------------------------------------------------------------------------
# Independent variables selection
X = dataset.iloc[:, [18,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,21,24,25,26,27]].values
X_valid = validation.iloc[:, [18,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,21,24,25,26,27]].values
# ABOVE we skip column index =1 to Avoid the Dummy Variable TRAP

# ----------------------------------------------------------------------------
# Dependent variable
y = dataset.iloc[:, 29].values
# SBASE40TH_LOC [CYCLE + 1 YEAR]


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


ctry_list = dataset.COMPCOUNTRY.unique()
filter_arr = []
filter_arr_valid = []
X_ctry = []
y_ctry = []
X_ctry_valid = []

for ctry in ctry_list:
    filter_arr = []
    filter_arr_valid = []
    X_ctry = []
    y_ctry = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    # TEST SET - PREDICTIONS
    for country in X[:,0]:
        if country == ctry:
            filter_arr.append(True)
        else:
            filter_arr.append(False)
    
       
    X_ctry = X[filter_arr]
    # We remove the country column from our table
    X_ctry = X_ctry[:,1:]
    y_ctry = y[filter_arr]
    
    # ----------------------------------------------------------------------------
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X_ctry, y_ctry, test_size = 0.2, random_state = 0)
    
    # ----------------------------------------------------------------------------
    # Fitting Multiple Linear Regression to the Training set
    regressor = LinearRegression()
    # By applying the fit method onto the 'regressor' object, it means I fit
    # the multiple Linear Regression to my training set
    regressor.fit(X_train, y_train)
    
    # vector of predictions
    y_pred = regressor.predict(X_test)
    
    # Differences
    difference = 100*abs(y_pred-y_test)/y_test
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    
    
    # TEST SET - TESTING
    # TEST SET - due to random split, data may come from 2018 or 2019, or mixed
    # TYPE a list of countries, which you are interested in
    if ctry in ['XX']:
        
        print('\n' + ctry+':\n' + 'AVG deviation = ' + str(difference.mean())) # Average deviation [percentage]
        print('Median of deviation = ' + str(np.median(difference))) # Median of deviation [percentage]
        print('Root-mean-square error = '+str(rmse))
        print('R^2 = ' + str(r2)+'\n')
        
        for i in range(0,15):
            if i==0:
                jfg = 'Aviation'
            elif i==1:
                jfg = 'Bus/Adm Svcs'
            elif i==2:
                jfg = 'Bus Mgmt and Develop'
            elif i==3:
                jfg = 'Finance'
            elif i==4:
                jfg = 'General Management'
            elif i==5:
                jfg = 'Human Resources'
            elif i==6:
                jfg = 'Info Technology'
            elif i==7:
                jfg = 'Legal'
            elif i==8:
                jfg = 'Marketing'
            elif i==9:
                jfg = 'Public Affairs'
            elif i==10:
                jfg = 'Retail and Attractns'
            elif i==11:
                jfg = 'Sales and Acct Mgmt'
            elif i==12:
                jfg = 'Security'
            elif i==13:
                jfg = 'Supply Chain'
            elif i==14:
                jfg = 'Technical'
            else:
                jfg = 'Credit Union'
                
            for j in range(0, len(X_test)):
                if X_test[j,i] == 1:
                    plt.scatter(X_test[j,15], y_test[j], color = 'red', label='Test data real scales')
                    plt.scatter(X_test[j,15], y_pred[j], alpha=0.5, color = 'blue', label='Test data predictions')
            plt.title(ctry + ' ' + jfg + ' Comparison: Real vs Predicted SBASE40TH LOCAL')
            plt.xlabel('Job Grade')
            plt.ylabel('SBASE40TH')
            plt.show()
            
            
            
    # TEST SET - TESTING previous year to predicted year -> INCREASE/DECREASE
    # TEST SET - due to random split, data may come from 2018 or 2019, or mixed
    # TYPE a list of countries, which you are interested in
    if ctry in ['XX']:
        
        for i in range(0,15):
            if i==0:
                jfg = 'Aviation'
            elif i==1:
                jfg = 'Bus/Adm Svcs'
            elif i==2:
                jfg = 'Bus Mgmt and Develop'
            elif i==3:
                jfg = 'Finance'
            elif i==4:
                jfg = 'General Management'
            elif i==5:
                jfg = 'Human Resources'
            elif i==6:
                jfg = 'Info Technology'
            elif i==7:
                jfg = 'Legal'
            elif i==8:
                jfg = 'Marketing'
            elif i==9:
                jfg = 'Public Affairs'
            elif i==10:
                jfg = 'Retail and Attractns'
            elif i==11:
                jfg = 'Sales and Acct Mgmt'
            elif i==12:
                jfg = 'Security'
            elif i==13:
                jfg = 'Supply Chain'
            elif i==14:
                jfg = 'Technical'
            else:
                jfg = 'Credit Union'
                
            for j in range(0, len(X_test)):
                if X_test[j,i] == 1:
                    plt.scatter(X_test[j,15], X_test[j,16], color = 'orange', label='Test data previous year scales')
                    plt.scatter(X_test[j,15], y_pred[j], alpha=0.5, color = 'blue', label='Test data predictions +1 year')
            plt.title(ctry + ' ' + jfg + ' Comparison: X Year Scales vs X+1 Year Predicted SBASE40TH LOCAL')
            plt.xlabel('Job Grade')
            plt.ylabel('SBASE40TH')
            plt.show()
    
    
    # VALIDATION SET - TESTING
    # VALIDATION SET - projecting 2020 scales to 2021
    # TYPE a list of countries, which you are interested in
    if ctry in ['XX']:
        for country in X_valid[:,0]:
            if country == ctry:
                filter_arr_valid.append(True)
            else:
                filter_arr_valid.append(False)
                
        X_ctry_valid = X_valid[filter_arr_valid]
        # We remove the country column from our table
        X_ctry_valid = X_ctry_valid[:,1:]
        
        y_pred_valid = regressor.predict(X_ctry_valid)
        
        for i in range(0,15):
            if i==0:
                jfg = 'Aviation'
            elif i==1:
                jfg = 'Bus/Adm Svcs'
            elif i==2:
                jfg = 'Bus Mgmt and Develop'
            elif i==3:
                jfg = 'Finance'
            elif i==4:
                jfg = 'General Management'
            elif i==5:
                jfg = 'Human Resources'
            elif i==6:
                jfg = 'Info Technology'
            elif i==7:
                jfg = 'Legal'
            elif i==8:
                jfg = 'Marketing'
            elif i==9:
                jfg = 'Public Affairs'
            elif i==10:
                jfg = 'Retail and Attractns'
            elif i==11:
                jfg = 'Sales and Acct Mgmt'
            elif i==12:
                jfg = 'Security'
            elif i==13:
                jfg = 'Supply Chain'
            elif i==14:
                jfg = 'Technical'
            else:
                jfg = 'Credit Union'
                
            for j in range(0, len(X_ctry_valid)):
                if X_ctry_valid[j,i] == 1:
                    plt.scatter(X_ctry_valid[j,15], X_ctry_valid[j,16], color = 'red', label='2020 scales')
                    plt.scatter(X_ctry_valid[j,15], y_pred_valid[j], alpha=0.5, color = 'blue', label='2021 scales predictions')
            plt.title(ctry + ' ' + jfg + ' 2020 projection to 2021 SBASE40TH LOCAL')
            plt.xlabel('Job Grade')
            plt.ylabel('SBASE40TH')
            plt.show()
















