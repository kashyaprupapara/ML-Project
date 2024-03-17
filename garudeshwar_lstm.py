# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import cross_val_score
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.wrappers.scikit_learn import KerasRegressor
#Reading CSV datafiles
dataset = pd.read_csv('garudeshwar_11.csv')
dataset = dataset.set_index(['Date'])
Train = dataset.iloc[:10958,:]
Test = dataset.iloc[10958:,:]
X_train = Train.iloc[:,:-1]
y_train = Train.iloc[:, -1]
X_test = Test.iloc[:,:-1]
y_test = Test.iloc[:, -1]
#split data
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
batch_size = [10,20]
epochs = [50,100]
optimizer = ['Adadelta', 'Adam']
activation = ['relu', 'tanh', 'sigmoid']
dropout_rate = [0.0, 0.2]
# Create the random grid
param_grid = {'batch_size': batch_size,
              'epochs': epochs,
              'optimizer': optimizer,
              'activation': activation,
              'dropout_rate': dropout_rate}
print(param_grid)
def create_model(activation, dropout_rate, optimizer):
 # create model
 model = Sequential()
 model.add(LSTM(4, input_shape=(3,1)))
 model.add(Dropout(dropout_rate))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', metrics=['accuracy'])
 return model
model = KerasRegressor(build_fn=create_model)
from sklearn.model_selection import GridSearchCV
model_Grid = GridSearchCV(estimator=model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
model_Grid.fit(X_train, y_train)
model_Grid.best_params_
best=model_Grid.best_params_
print(model_Grid.best_params_)
m1 = create_model(n_estimators=best['n_estimators'], 
                                          bootstrap=best['bootstrap'], 
                                          max_depth=best['max_depth'], 
                                          max_features=best['max_features'], 
                                          min_samples_leaf=best['min_samples_leaf'], 
                                          min_samples_split=best['min_samples_split'])  
m1.fit(X_train, y_train)
y_pred = m1.predict(X_test)
y_pred_1 = m1.predict(X_train)
y_pred=(y_pred*2210.308)+784.8977
y_pred_1=(y_pred_1*2210.308)+784.8977
y_test=(y_test*2210.308)+784.8977 
y_train=(y_train*2210.308)+784.8977
from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error 1:',metrics.mean_absolute_error(y_train, y_pred_1))
print('Mean Squared Error 1:',metrics.mean_squared_error(y_train, y_pred_1))
print('Root Mean Squared Error 1:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_1)))
print('r2:',metrics.r2_score(y_test, y_pred))
print('r2:',metrics.r2_score(y_train, y_pred_1)) 
plt.rcParams['figure.figsize']=(12,8)
x_ax=range(len(X_test))
plt.plot(x_ax, y_test, label='observed', color='k', linestyle='-')
plt.plot(x_ax, y_pred, label='predicted', color='g', linestyle='--') 
plt.show()
plt.rcParams['figure.figsize']=(16,8)
x_ax=range(len(X_train))
plt.plot(x_ax, y_train, label='observed', color='k', linestyle='-')
plt.plot(x_ax, y_pred_1, label='predicted', color='r', linestyle='--') 
plt.show()
df = pd.DataFrame(y_pred)
df.to_csv("gd_pred9.csv")
df1 = pd.DataFrame(y_pred_1)
df1.to_csv("gd_pred_19.csv")
y_test.to_csv("gd_test9.csv")
y_train.to_csv("gd_train9.csv")