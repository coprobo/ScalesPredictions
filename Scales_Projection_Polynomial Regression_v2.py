# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:03:39 2020

@author: O46743 - Piotr Walędzik
"""



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('.\\3. Data Pre-Processed\\SCALES_Pre-Processed_POLAND.csv')

# Dataset has been pre-processed in Alteryx


# ----------------------------------------------------------------------------
# Independent variables selection
X = dataset.iloc[:, [4,10,5,3]].values
# 10 - Scales Year | 5 - Market Movement | 2 - Country
# 4 - Job Family Group | 3 - Grade

# ----------------------------------------------------------------------------
# Dependent variable
y = dataset.iloc[:, 7].values
# SBASE40TH_LOC

# ----------------------------------------------------------------------------
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# '0' below tells the encoder which column we want to encode into a dummy variable
# since in our table, the 'JOBFAMGROUP" is under 0th column [0 indexed -> 0,1,2,3...]
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X).toarray(), dtype=np.float) # no idea why I had to put:
# .toarray() ... in another model works fine without this piece :(


# now our X has the JOBFAMGROUP encoded into dummy data of 0s and 1s


# Avoiding the Dummy Variable TRAP
X = X[:, 1:] # we remove one dummy variable-> to avoid dummy variable trap


# ----------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)


# ----------------------------------------------------------------------------
# Fitting Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# 'degree' specifies the degree of polynomial features
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)


# vector of predictions
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))


# Differences
difference = 100*abs(y_pred-y_test)/y_test
print('\n AVG deviation = ' + str(difference.mean())) # Average deviation [percentage]
print('\n Median of deviation = ' + str(np.median(difference))) # Median of deviation [percentage]

from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print('\n root-mean-square error = '+str(rmse))
print('\n R^2 = ' + str(r2))

# Visualising the Polynomial Regression results for each Job Family Group separately
# in 2 Dimensions only

for i in range(0,14):
    for j in range(0, len(X_test)):
        if X_test[j,i] == 1:
            plt.scatter(X_test[j,16], y_test[j], color = 'red', label='Test data real scales')
            plt.scatter(X_test[j,16], y_pred[j], alpha=0.5, color = 'blue', label='Test data predictions')
    plt.title('Predictions of SBASE40TH LOCAL iter = ' + str(i))
    plt.xlabel('Job Grade')
    plt.ylabel('SBASE40TH')
    plt.show()


# TESTING FOR 2021
X_2021 = np.copy(X[np.where(X[:,14]==2020)])
X_2021[:,14] = X_2021[:,14]+1 #Let's test for year 2021
X_2021[:,15] = X_2021[:,15]+1.5 # MKT MOV increase

y_2021 = lin_reg_2.predict(poly_reg.fit_transform(X_2021))

for i in range(0,14):
    for j in range(0, len(X_test)):
        if X_2021[j,i] == 1:
            plt.scatter(X_2021[j,16], y_2021[j], color = 'purple', label='2021 predictions')
    plt.title('2021 Predictions of SBASE40TH LOCAL iter = ' + str(i))
    plt.xlabel('Job Grade')
    plt.ylabel('SBASE40TH')
    plt.show()











# HOW GOOD OUR MODEL IS ?
import statsmodels.api as sm
# Polynomial Regression has its x0 ommitted in the dataset because it is always =1
# however for 'Backward Elimination' using 'statsmodels' package
# we need to add that column of 1s to our data model, to mimic x0=1
X_validate = np.append(arr = np.ones(shape=(len(X_test),1)).astype(int), values=X_test, axis =1)


# 1. Select a significance level to stay in the model (e.g. SL=0.05)
# if a P-Value of an independent variable is above SL (P-Value > SL),
# then we will remove that variable, else it stays in the model

# 2. Fit the full model with all possible predictors
# OLS = Ordinary Least Squares method
regressor_OLS = sm.OLS(endog = y_test, exog = X_validate).fit()

# 3. We print the summary tale to read which variable has the highest P-Value
regressor_OLS.summary()