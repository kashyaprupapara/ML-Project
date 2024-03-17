# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import cross_val_score
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
#import scikeras
#from scikeras.wrappers import KerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
for i in range(1,9):
    if i==1:
        #Reading CSV datafiles
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\ghala_18.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7305,:]
        Test = dataset.iloc[7305:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        # Create the random grid
        def create_model(neurons=12, dropout_rate=0.2, optimizer='Adam', activation='relu'):
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, activation='relu'))
            model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
            return model
        model = KerasRegressor(build_fn=create_model, neurons=12, dropout_rate=0.2, 
                               optimizer='Adam', activation='relu')
        best_model = create_model(neurons=12, 
                                  dropout_rate=0.2, 
                                  optimizer='Adam',
                                  activation='relu')
        best_model.fit(X_train, y_train, epochs=20, batch_size=10)
        y_pred = best_model.predict(X_test)
        y_pred_1 = best_model.predict(X_train)
        #y_pred=(y_pred*818.2461)+164.3152
        #y_pred_1=(y_pred_1*818.2461)+164.3152
        #y_test=(y_test*818.2461)+164.3152
        #y_train=(y_train*818.2461)+164.3152
        #y_pred=(y_pred*2210.308)+784.8977
        #y_pred_1=(y_pred_1*2210.308)+784.8977
        #y_test=(y_test*2210.308)+784.8977 
        #y_train=(y_train*2210.308)+784.8977
        y_pred=(y_pred*453.9898)+164.11
        y_pred_1=(y_pred_1*453.9898)+164.11
        y_test=(y_test*453.9898)+164.11
        y_train=(y_train*453.9898)+164.11
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
        #df = pd.DataFrame(y_pred)
        #df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\khanpur\\k\\rnn\\D\\klstm_pred91.csv")
        #df1 = pd.DataFrame(y_pred_1)
        #df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\khanpur\\k\\rnn\\D\\klstm_pred_191.csv")
        #y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\khanpur\\k\\rnn\\D\\klstm_test2.csv")
        #y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\khanpur\\k\\rnn\\D\\klstm_train2.csv")
        #df = pd.DataFrame(y_pred)
        #df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\garudeshwar\\gd\\rnn\\D\\gdlstm_pred92.csv")
        #df1 = pd.DataFrame(y_pred_1)
        #df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\garudeshwar\\gd\\rnn\\D\\gdlstm_pred_192.csv")
        #y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\khanpur\\k\\rnn\\D\\klstm_test2.csv")
        #y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\khanpur\\k\\rnn\\D\\klstm_train2.csv")
        df = pd.DataFrame(y_pred)
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\lstm\\D\\glstm_pred8.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\lstm\\D\\glstm_pred_18.csv")
    elif i==2:
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\practice.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:2557,:]
        Test = dataset.iloc[2557:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        def create_model(neurons=16, dropout_rate=0.2, optimizer='SGD', activation='tanh'):
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, activation='tanh'))
            model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])
            return model
        model = KerasRegressor(build_fn=create_model, neurons=16, dropout_rate=0.2, 
                               optimizer='SGD', activation='tanh')
        best_model = create_model(neurons=16, 
                                  dropout_rate=0.2, 
                                  optimizer='SGD',
                                  activation='tanh')
        best_model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))
        y_pred = best_model.predict(X_test)
        y_pred_1 = best_model.predict(X_train)
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
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\garudeshwar\\gd\\rnn\\gdlstm_pred9.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\garudeshwar\\gd\\rnn\\gdlstm_pred_19.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\garudeshwar\\gd\\rnn\\gdlstm_test9.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\garudeshwar\\gd\\rnn\\gdlstm_train9.csv")
    elif i==3:
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\ghala_11.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7305,:]
        Test = dataset.iloc[7305:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        batch_size = [20, 50]
        epochs = [20, 50]
        optimizer = ['Adadelta', 'Adam']
        activation = ['relu', 'tanh', 'softmax']
        dropout_rate = [0.0, 0.2]
        neurons = [16, 64]
        # Create the random grid
        param_grid = {'batch_size': batch_size,
                      'epochs': epochs,
                      'optimizer': optimizer,
                      'activation': activation,
                      'dropout_rate': dropout_rate,
                      'neurons': neurons}
        print(param_grid)
        def create_model(neurons=16, dropout_rate=0.2, optimizer='adam', activation='relu'):
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, activation='relu'))
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
            return model
        model = KerasRegressor(build_fn=create_model, neurons=16, dropout_rate=0.2, optimizer='adam', activation='relu')
        from sklearn.model_selection import GridSearchCV
        model_Grid = GridSearchCV(estimator=model, param_grid = param_grid, cv = 2, verbose=0)
        model_Grid.fit(X_train, y_train)
        model_Grid.best_params_
        best=model_Grid.best_params_
        print(model_Grid.best_params_)
        best_model = create_model(neurons=best['neurons'], 
                                  dropout_rate=best['dropout_rate'], 
                                  optimizer=best['optimizer'],
                                  activation=best['activation'])
        best_model.fit(X_train, y_train, epochs=best['epochs'], batch_size=best['batch_size'], validation_data=(X_test, y_test))
        y_pred = best_model.predict(X_test)
        y_pred_1 = best_model.predict(X_train)
        y_pred=(y_pred*453.9898)+164.11
        y_pred_1=(y_pred_1*453.9898)+164.11
        y_test=(y_test*453.9898)+164.11
        y_train=(y_train*453.9898)+164.11
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
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\lstm\\y_pred1.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\lstm\\y_pred_11.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\lstm\\y_test1.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\lstm\\y_train1.csv")