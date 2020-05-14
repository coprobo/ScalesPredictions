# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:03:39 2020

@author: O46743
"""



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
print(difference.mean()) # Average deviation [percentage]
print(np.median(difference)) # Median of deviation [percentage]


# Visualising the Polynomial Regression results for each Job Family Group separately
# in 2 Dimensions only

for i in range(0,12):
    for j in range(0, len(X_test)):
        if X_test[j,i] == 1:
            plt.scatter(X_test[j,12], y_test[j], color = 'red')
            plt.scatter(X_test[j,12], y_pred[j], alpha=0.5, color = 'blue')
    plt.title('Predictions of SBASE40TH LOCAL iter = ' + str(i))
    plt.xlabel('Job Grade')
    plt.ylabel('SBASE40TH')
    plt.show()
