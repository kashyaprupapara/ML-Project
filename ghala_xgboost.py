# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
for i in range(1,9):
    if i==1:
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\ghala_19.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7305,:]
        Test = dataset.iloc[7305:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        model = xgb.XGBRegressor()
        # Define the hyperparameters
        params = {
            'max_depth': [None],
            'learning_rate': [0.01],
            'n_estimators': [200],
            'gamma': [0.1],
            'reg_alpha': [0.1],
            'reg_lambda': [1.5]
        }
        from sklearn.model_selection import GridSearchCV
        # Perform GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(model, params, n_jobs=1, cv=3, verbose=1)
        grid_search.fit(X_train, y_train)
        # Print the best hyperparameters
        print("Best hyperparameters: ", grid_search.best_params_)
        # Train the model with the best hyperparameters
        best_model = xgb.XGBRegressor(**grid_search.best_params_)
        best_model.fit(X_train, y_train)
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
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\xgboost\\y_pred8.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\xgboost\\y_pred_18.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\xgboost\\y_test1.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\xgboost\\y_train1.csv")