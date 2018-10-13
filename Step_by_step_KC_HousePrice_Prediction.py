# -*- coding: utf-8 -*-
"""
Linear Regrassion Models to Predict KC House Prices
"""

#import Modules
import numpy as np
import pandas as pd
from sklearn import linear_model,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#importing Data in to Data frame
df=pd.read_csv('kc_house_data.csv')
print(df.head())
df.dtypes

X=df.iloc[:,2:-1].values
Y=df.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 5] = labelencoder.fit_transform(X[:, 5])
X[:, 7] = labelencoder.fit_transform(X[:, 7])
#visualizing data with correlation analysis
sns.heatmap(df.corr(),cmap='Blues',annot=True)

plt.figure(figsize=(20,30))
mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(df.corr(), mask=mask, vmax=.2, square=True)
plt.show()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)



'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)
'''

lr=linear_model.LinearRegression()
lr.fit(X_train,Y_train)
Y_red=lr.predict(X_test)



lr.coef_

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((21613,36)).astype(int), values = X, axis = 1)
X_opt=X[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
print(lr.score(Y_red,Y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_red))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_red))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_red)))

from sklearn.model_selection import cross_val_score
cross_val_score(lr,X_train,Y_train)
