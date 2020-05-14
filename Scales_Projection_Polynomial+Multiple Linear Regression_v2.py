# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:54:57 2020

@author: O46743
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('.\\3. Data Pre-Processed\\v2\\SCALES_Pre-Processed_POLAND_HR_training_test.csv') # TYLKO HR DLA POLSKI!!!!

validation = pd.read_csv('.\\3. Data Pre-Processed\\v2\\SCALES_Pre-Processed_POLAND_HR_validation.csv')

# Dataset has been pre-processed in Alteryx

# ----------------- FOR MYSELF - GOOD DATA FILLER ----------------------------
# Filling the Missing Values with MEDIAN of each group
# First with Median achieved by groupping by Country Code, Job Family Gr., JG

# dataset["base_salary_75th [LOCAL]"] = dataset.\
#     groupby(by = ['COUNTRY_CODE','Job Family Group','JG'])["base_salary_75th [LOCAL]"].\
#     transform(lambda x: x.fillna(x.median()))


# ----------------------------------------------------------------------------
# Independent variables selection
X = dataset.iloc[:, [2,5,8]].values
# 0 - Scales Year | 2 - JG | 3 - JFG | 5 - SBASE40TH_LOC [CYCLE YEAR]
# 8 - MKTMOV | 9 - GDP | 10 - Inflation | 11 - Unemployment --> ALL ARE [CYCLE YEAR]

# !!! NOT USED: 3 - JFG, 0 - Scales Year

# ----------------------------------------------------------------------------
# Dependent variable
y = dataset.iloc[:, 13].values
# SBASE40TH_LOC [CYCLE + 1 YEAR]

# ----------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


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


# 2D Training values
plt.scatter(X_train[:,0], y_train, color = 'orange', label='2018+2019 Training values')
plt.title('SBASE40TH LOCAL Training values')
plt.xlabel('Job Grade')
plt.ylabel('SBASE40TH')
plt.legend()
plt.show()

# 2D Test values
plt.scatter(X_test[:,0], y_test, color = 'purple', label='2018+2019 Test values')
plt.title('SBASE40TH LOCAL Test values - NO PREDICTIONS')
plt.xlabel('Job Grade')
plt.ylabel('SBASE40TH')
plt.legend()
plt.show()


# 2D Training + Test
plt.scatter(X_test[:,0], y_test, color = 'purple', label='2018+2019 Test values')
plt.scatter(X_train[:,0], y_train, color = 'orange', label='2018+2019 Training values')
plt.title('SBASE40TH LOCAL training vs Test - GOOD no outliers')
plt.xlabel('Job Grade')
plt.ylabel('SBASE40TH')
plt.legend()
plt.show()


# 3D plot for 2018+2019 predictions
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs1 = X_test[:,0]
ys1 = X_test[:,2]
zs1 = y_test

zs2 = y_pred

ax.scatter(xs1, ys1, zs1, marker='o')
ax.scatter(xs1, ys1, zs2, marker='^')
ax.set_xlabel('JG')
ax.set_ylabel('MKTMOV')
ax.set_zlabel('Real/Predicted TEST values')
plt.show()




# TESTING CURRENT SCALES 2020 predicted to 2021
X_valid = validation.iloc[:, [2,5,8]].values
y_valid_pred = lin_reg_2.predict(poly_reg.fit_transform(X_valid))

plt.scatter(X_valid[:,0], y_valid_pred, color = 'purple', label='2021 predicted values')
plt.title('SBASE40TH LOCAL 2020 Validation set predicted values for 2021')
plt.xlabel('Job Grade')
plt.ylabel('SBASE40TH')
plt.legend()
plt.show()


# Comparison of how 2020 were predicted to 2021
plt.scatter(X_valid[:,0], y_valid_pred, color = 'orange', label='2021 predicted values')
plt.scatter(X_valid[:,0], X_valid[:,1], color = 'black',alpha=0.5, label='2020 real values')
plt.title('Comparison: SBASE40TH LOCAL 2020 Validation set projected to 2021')
plt.xlabel('Job Grade')
plt.ylabel('SBASE40TH')
plt.legend()
plt.show()


# 3D plot for 2021 predictions
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = X_valid[:,0]
ys = X_valid[:,2]
zs = y_valid_pred
ax.scatter(xs, ys, zs, marker='o')
# ax.plot(xs, ys, zs)
ax.set_xlabel('JG')
ax.set_ylabel('MKTMOV')
ax.set_zlabel('2021 predicted values')
plt.show()


# PROBLEMS WITH MULTIVARIATE POLYNOMIAL
# REGRESSION
# 1) The major issue with Multivariate Polynomial Regression is
# the problem of Multicolinearity. When there are multiple regression variables, there are high chances that the variables
# are interdependent on each other. In such cases, due to this
# this relationship amongst variables, the regression equation
# computed does not properly fit the original graph.
# 2) Another problem with Multivariate Polynomial Regression
# is that the higher degree terms in the equation do not contribute majorly to the regression equation. So they can be ignored.
# But if the degree is each time estimated and decided, if required
# or not, then each time all the parameters and equations need
# to be computed.
# PAPER: https://www.ijser.org/researchpaper/Multivariate-Polynomial-Regression-in-Data-Mining-Methodology.pdf




# ################################################################################################################################



# MULTIPLE LINEAR REGRESSION FOR WHOLE COUNTRY=POLAND !!!!!!!


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('.\\3. Data Pre-Processed\\v2\\SCALES_Pre-Processed_POLAND_training_test.csv') # TO JEST CAŁA POLSKA - WSZYSTKIE JFG

# Dataset has been pre-processed in Alteryx


# ----------------------------------------------------------------------------
# Independent variables selection
X = dataset.iloc[:, [3,2,5,8,9,10,11]].values
# 0 - Scales Year | 2 - JG | 3 - JFG | 5 - SBASE40TH_LOC [CYCLE YEAR]
# 8 - MKTMOV | 9 - GDP | 10 - Inflation | 11 - Unemployment --> ALL ARE [CYCLE YEAR]


# ----------------------------------------------------------------------------
# Dependent variable
y = dataset.iloc[:, 13].values
# SBASE40TH_LOC [CYCLE + 1 YEAR]


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

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

for i in range(0,12):
    if i==0:
        jfg = 'Bus/Adm Svcs'
    elif i==1:
        jfg = 'Finance'
    elif i==2:
        jfg = 'General Management'
    elif i==3:
        jfg = 'Human Resources'
    elif i==4:
        jfg = 'Info Technology'
    elif i==5:
        jfg = 'Legal'
    elif i==6:
        jfg = 'Marketing'
    elif i==7:
        jfg = 'Public Affairs'
    elif i==8:
        jfg = 'Sales and Acct Mgmt'
    elif i==9:
        jfg = 'Security'
    elif i==10:
        jfg = 'Supply Chain'
    elif i==11:
        jfg = 'Technical'
    else:
        jfg = 'Bus Mgmt and Develop'
        
    for j in range(0, len(X_test)):
        if X_test[j,i] == 1:
            plt.scatter(X_test[j,12], y_test[j], color = 'red', label='Test data real scales')
            plt.scatter(X_test[j,12], y_pred[j], alpha=0.5, color = 'blue', label='Test data predictions')
    plt.title(jfg + ' Comparison: Real vs Predicted SBASE40TH LOCAL iter = ' + str(i))
    plt.xlabel('Job Grade')
    plt.ylabel('SBASE40TH')
    plt.show()



