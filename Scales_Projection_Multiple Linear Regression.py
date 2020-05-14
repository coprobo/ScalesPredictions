# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:44:01 2020

@author: O46743
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('Market_Data_Pre-Processed_POLAND.csv')
# Dataset has been pre-processed in Alteryx
# LEFT+RIGHT combined together
# Weight > 0
# Anchor Jobs ONLY

# ----------------------------------------------------------------------------
# Filling the Missing Values with MEDIAN of each group
# First with Median achieved by groupping by Country Code, Job Family Gr., JG
# If still NAs exist we fill them with values achieved from groupping by Country Code, JG
dataset["base_salary_75th [LOCAL]"] = dataset.\
    groupby(by = ['COUNTRY_CODE','Job Family Group','JG'])["base_salary_75th [LOCAL]"].\
    transform(lambda x: x.fillna(x.median()))

dataset["base_salary_75th [LOCAL]"] = dataset.\
    groupby(by = ['COUNTRY_CODE','JG'])["base_salary_75th [LOCAL]"].\
        transform(lambda x: x.fillna(x.median()))
 
# ************************OLD SOLUTION****************************************       
# OLD APPROACH for filling the missing data - VERY INEFFICIENT
# median_JFG_JG = dataset.groupby(by =['COUNTRY_CODE','Job Family Group','JG'], as_index=False).median()
# median_JG = dataset.groupby(by =['COUNTRY_CODE','JG'], as_index = False).median()

# for index, row in dataset.iterrows():
#     if pd.isnull(row['base_salary_75th [LOCAL]']):
#         ctry_cd = row['COUNTRY_CODE']
#         jfg = row['Job Family Group']
#         jg = row['JG']
#         for index2, row2 in median_JFG_JG.iterrows():
#             if pd.isnull(row2['base_salary_75th [LOCAL]'])==False and row2['COUNTRY_CODE']==ctry_cd and row2['Job Family Group']==jfg and row2['JG']==jg:
#                 dataset.loc[index,'base_salary_75th [LOCAL]'] = median_JFG_JG.loc[index2, 'base_salary_75th [LOCAL]']
#             else:
#                 for index3, row3 in median_JG.iterrows():
#                     if pd.isnull(row3['base_salary_75th [LOCAL]'])==False and row3['COUNTRY_CODE']==ctry_cd and row3['JG']==jg:
#                         dataset.loc[index,'base_salary_75th [LOCAL]'] = median_JG.loc[index3, 'base_salary_75th [LOCAL]']
# ****************************************************************************

# ----------------------------------------------------------------------------
# Independent variables selection
X = dataset.iloc[:, [7,2,5,12,14]].values
# 7 - Job Family Group | 2 - Market Movement | 5 - Compensation Grade
# 12 - 50th Raw Market data | 14 - 75th Raw Market data

# ----------------------------------------------------------------------------
# Dependent variable
y = dataset.iloc[:, 35].values
# SBASE40TH_LOC

# ----------------------------------------------------------------------------
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# '0' below tells the encoder which column we want to encode into a dummy variable
# since in our table, the 'JOBFAMGROUP" is under 0th column [0 indexed -> 0,1,2,3...]
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# now our X has the JOBFAMGROUP encoded into dummy data of 0s and 1s

# Avoiding the Dummy Variable TRAP
X = X[:, 1:] # we remove one dummy variable-> to avoid dummy variable trap

# ----------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# ----------------------------------------------------------------------------
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
# By applying the fit method onto the 'regressor' object, it means I fit
# the multiple Linear Regression to my training set
regressor.fit(X_train, y_train)



# TEST PHASE !!!!!!!!!!!!!
# ----------------------------------------------------------------------------
# Predicting the Test set results

# vector of predictions
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination
import statsmodels.api as sm
# Multiple Linear Regression model is y =b0x0 + b1x1 + ... + bnxn
# where normally x0 is ommitted in the dataset because it is always =1
# however for 'Backward Elimination' using 'statsmodels' package
# we need to add that column of 1s to our data model, to mimic x0=1
X = np.append(arr = np.ones(shape=(len(X),1)).astype(int), values=X, axis =1)

# Backward Elimination
# X_opt will hold only statistically significant variables
# We start checking if the variables are significat first
# That;s why we need to import all the variables at the beginning
# which are 0,1,2,3,4,5 columns from X set
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]

# 1. Select a significance level to stay in the model (e.g. SL=0.05)
# if a P-Value of an independent variable is above SL (P-Value > SL),
# then we will remove that variable, else it stays in the model

# 2. Fit the full model with all possible predictors
# OLS = Ordinary Least Squares method
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# 3. We print the summary tale to read which variable has the highest P-Value
regressor_OLS.summary()




# REPEAT ALL THE STEPS
X_opt = X[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# REPEAT ALL THE STEPS
X_opt = X[:,[0,2,4,5,6,7,8,9,10,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# REPEAT ALL THE STEPS
X_opt = X[:,[0,2,4,5,7,8,9,10,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# REPEAT ALL THE STEPS
X_opt = X[:,[0,2,4,5,7,8,10,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# REPEAT ALL THE STEPS
X_opt = X[:,[0,2,4,5,7,8,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# REPEAT ALL THE STEPS
X_opt = X[:,[0,2,4,7,8,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# REPEAT ALL THE STEPS
X_opt = X[:,[0,2,4,7,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# REPEAT ALL THE STEPS
X_opt = X[:,[0,2,4,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# REPEAT ALL THE STEPS
X_opt = X[:,[0,2,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # ten model ma Adj. R-quared = 0.932


# REPEAT ALL THE STEPS
X_opt = X[:,[0,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # ten model ma mniejszy Adj. R-squared = 0.931 --> dlatego wybieram jeden model wczesniej



# Always look at 'Adj. R-squared' metric, which should be the highest in the
# selected model (closest to 1), if it starts dropping with the next step it means that 
# previous model was better fit


# How to interpret the 'COEF''
# 'coef' - coefficients of the model, if they are positive it means
# if the independent variable for that coefficient grows, the dependent variable
# will grow as well





# MODEL RE-DONE

# ----------------------------------------------------------------------------
# Independent variables selection
X = dataset.iloc[:, [7,2,5,12,14]].values
# 7 - Job Family Group | 2 - Market Movement | 5 - Compensation Grade
# 12 - 50th Raw Market data | 14 - 75th Raw Market data

# ----------------------------------------------------------------------------
# Dependent variable
y = dataset.iloc[:, 35].values
# SBASE40TH_LOC

# ----------------------------------------------------------------------------
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# '0' below tells the encoder which column we want to encode into a dummy variable
# since in our table, the 'JOBFAMGROUP" is under 0th column [0 indexed -> 0,1,2,3...]
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# now our X has the JOBFAMGROUP encoded into dummy data of 0s and 1s

# Avoiding the Dummy Variable TRAP
X = X[:, 1:] # we remove one dummy variable-> to avoid dummy variable trap



# Adjusting to the tested model from above
# X_opt = X[:,[0,2,11,12,13,14,15]]
X = X[:,[1,10,11,12,13,14]]



# ----------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# ----------------------------------------------------------------------------
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
# By applying the fit method onto the 'regressor' object, it means I fit
# the multiple Linear Regression to my training set
regressor.fit(X_train, y_train)



# TEST PHASE !!!!!!!!!!!!!
# ----------------------------------------------------------------------------
# Predicting the Test set results

# vector of predictions
y_pred = regressor.predict(X_test)


# Visualising the Multiple Linear Regression result
plt.scatter(X_test[:,3], y_test, color = 'red')
plt.plot(X_test[:,3], y_pred, color = 'blue')
plt.title('Predictions of SBASE40TH LOCAL')
plt.xlabel('Job Grade')
plt.ylabel('SBASE40TH')
plt.show()



