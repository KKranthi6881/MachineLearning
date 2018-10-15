# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 00:06:52 2018

@author: kkondapaka
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
import xgboost
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#importing Data in to Data frame
df=pd.read_csv('kc_house_data.csv')
print(df.head())
df.dtypes

X=df.loc[:,['bedrooms','bathrooms','floors','zipcode','yr_built','yr_renovated','condition','grade','waterfront', 'view','long','sqft_above']].values
Y=df.loc[:,['price']].values
#df_X=df.loc[:,['bedrooms','bathrooms','floors','zipcode','yr_built','yr_renovated','condition','grade','waterfront', 'view']]


scaler = StandardScaler()
X=scaler.fit_transform(X)
Y=scaler.fit_transform(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)



from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import regularizers

	# create model
model = Sequential()
model.add(Dense(320, input_dim=12, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(p = 0.1))
model.add(Dense(320, input_dim=320, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(p = 0.1))
model.add(Dense(160, input_dim=320, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(160, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['accuracy'])
	

history = model.fit(X_train, Y_train, epochs=200, batch_size=100,validation_data=(X_test, Y_test))

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


Y_pred=model.predict(X_test)
print(explained_variance_score(Y_pred,Y_test))
