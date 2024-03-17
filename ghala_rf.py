# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
for i in range(1,9):
    if i==1:
        #Reading CSV datafiles
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\ghala_19.csv')
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
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [100]
        # Number of features to consider at every split
        max_features = ['auto']
        # Maximum number of levels in tree
        max_depth = [30]
        #max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [4]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [5]
        # Method of selecting samples for training each tree
        bootstrap = [True]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(param_grid)
        rf_Model = RandomForestRegressor()
        from sklearn.model_selection import GridSearchCV
        rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
        rf_Grid.fit(X_train, y_train)
        rf_Grid.best_params_
        best=rf_Grid.best_params_
        print(rf_Grid.best_params_)
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=best['n_estimators'], 
                                          bootstrap=best['bootstrap'], 
                                          max_depth=best['max_depth'], 
                                          max_features=best['max_features'], 
                                          min_samples_leaf=best['min_samples_leaf'], 
                                          min_samples_split=best['min_samples_split'])  
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_pred_1 = regressor.predict(X_train)
        import statistics
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
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\y_pred_9.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\y_pred_1_9.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\y_test9.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala\\g\\y_train9.csv") 
    elif i==2:
        #Reading CSV datafiles
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala_12.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7306,:]
        Test = dataset.iloc[7306:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [50,100,200,300]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [5,10,30]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2,3,4,5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,3,5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(param_grid)
        rf_Model = RandomForestRegressor()
        from sklearn.model_selection import GridSearchCV
        rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
        rf_Grid.fit(X_train, y_train)
        rf_Grid.best_params_
        best=rf_Grid.best_params_
        print(rf_Grid.best_params_)
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=best['n_estimators'], 
                                          bootstrap=best['bootstrap'], 
                                          max_depth=best['max_depth'], 
                                          max_features=best['max_features'], 
                                          min_samples_leaf=best['min_samples_leaf'], 
                                          min_samples_split=best['min_samples_split']) 
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_pred_1 = regressor.predict(X_train)
        y_pred=(y_pred*389.5548)+127.4741
        y_pred_1=(y_pred_1*389.5548)+127.4741
        y_test=(y_test*389.5548)+127.4741
        y_train=(y_train*389.5548)+127.4741
        from sklearn import metrics
        print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('Mean Absolute Error 1:',metrics.mean_absolute_error(y_train, y_pred_1))
        print('Mean Squared Error 1:',metrics.mean_squared_error(y_train, y_pred_1))
        print('Root Mean Squared Error 1:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_1)))
        print('r2:',metrics.r2_score(y_test, y_pred))
        print('r2:',metrics.r2_score(y_train, y_pred_1))
        df = pd.DataFrame(y_pred)
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred2.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred_12.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_test2.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_train2.csv")
    elif i==3:
        #Reading CSV datafiles
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala_13.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7306,:]
        Test = dataset.iloc[7306:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [50,100,200,300]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [5,10,30]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2,3,4,5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,3,5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(param_grid)
        rf_Model = RandomForestRegressor()
        from sklearn.model_selection import GridSearchCV
        rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
        rf_Grid.fit(X_train, y_train)
        rf_Grid.best_params_
        best=rf_Grid.best_params_
        print(rf_Grid.best_params_)
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=best['n_estimators'], 
                                          bootstrap=best['bootstrap'], 
                                          max_depth=best['max_depth'], 
                                          max_features=best['max_features'], 
                                          min_samples_leaf=best['min_samples_leaf'], 
                                          min_samples_split=best['min_samples_split']) 
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_pred_1 = regressor.predict(X_train)
        y_pred=(y_pred*389.5548)+127.4741
        y_pred_1=(y_pred_1*389.5548)+127.4741
        y_test=(y_test*389.5548)+127.4741
        y_train=(y_train*389.5548)+127.4741
        from sklearn import metrics
        print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('Mean Absolute Error 1:',metrics.mean_absolute_error(y_train, y_pred_1))
        print('Mean Squared Error 1:',metrics.mean_squared_error(y_train, y_pred_1))
        print('Root Mean Squared Error 1:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_1)))
        print('r2:',metrics.r2_score(y_test, y_pred))
        print('r2:',metrics.r2_score(y_train, y_pred_1)) 
        df = pd.DataFrame(y_pred)
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred3.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred_13.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_test3.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_train3.csv")
    elif i==4:
        #Reading CSV datafiles
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala_14.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7306,:]
        Test = dataset.iloc[7306:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [50,100,200,300]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [5,10,30]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2,3,4,5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,3,5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(param_grid)
        rf_Model = RandomForestRegressor()
        from sklearn.model_selection import GridSearchCV
        rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
        rf_Grid.fit(X_train, y_train)
        rf_Grid.best_params_
        best=rf_Grid.best_params_
        print(rf_Grid.best_params_)
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=best['n_estimators'], 
                                          bootstrap=best['bootstrap'], 
                                          max_depth=best['max_depth'], 
                                          max_features=best['max_features'], 
                                          min_samples_leaf=best['min_samples_leaf'], 
                                          min_samples_split=best['min_samples_split']) 
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_pred_1 = regressor.predict(X_train)
        y_pred=(y_pred*389.5548)+127.4741
        y_pred_1=(y_pred_1*389.5548)+127.4741
        y_test=(y_test*389.5548)+127.4741
        y_train=(y_train*389.5548)+127.4741
        from sklearn import metrics
        print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('Mean Absolute Error 1:',metrics.mean_absolute_error(y_train, y_pred_1))
        print('Mean Squared Error 1:',metrics.mean_squared_error(y_train, y_pred_1))
        print('Root Mean Squared Error 1:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_1)))
        print('r2:',metrics.r2_score(y_test, y_pred))
        print('r2:',metrics.r2_score(y_train, y_pred_1)) 
        df = pd.DataFrame(y_pred)
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred4.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred_14.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_test4.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_train4.csv")
    elif i==5:
        #Reading CSV datafiles
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala_15.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7306,:]
        Test = dataset.iloc[7306:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [50,100,200,300]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [5,10,30]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2,3,4,5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,3,5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(param_grid)
        rf_Model = RandomForestRegressor()
        from sklearn.model_selection import GridSearchCV
        rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
        rf_Grid.fit(X_train, y_train)
        rf_Grid.best_params_
        best=rf_Grid.best_params_
        print(rf_Grid.best_params_)
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=best['n_estimators'], 
                                          bootstrap=best['bootstrap'], 
                                          max_depth=best['max_depth'], 
                                          max_features=best['max_features'], 
                                          min_samples_leaf=best['min_samples_leaf'], 
                                          min_samples_split=best['min_samples_split']) 
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_pred_1 = regressor.predict(X_train)
        y_pred=(y_pred*389.5548)+127.4741
        y_pred_1=(y_pred_1*389.5548)+127.4741
        y_test=(y_test*389.5548)+127.4741
        y_train=(y_train*389.5548)+127.4741
        from sklearn import metrics
        print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('Mean Absolute Error 1:',metrics.mean_absolute_error(y_train, y_pred_1))
        print('Mean Squared Error 1:',metrics.mean_squared_error(y_train, y_pred_1))
        print('Root Mean Squared Error 1:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_1)))
        print('r2:',metrics.r2_score(y_test, y_pred))
        print('r2:',metrics.r2_score(y_train, y_pred_1)) 
        df = pd.DataFrame(y_pred)
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred5.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred_15.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_test5.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_train5.csv")
    elif i==6:
        #Reading CSV datafiles
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala_16.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7306,:]
        Test = dataset.iloc[7306:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [50,100,200,300]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [5,10,30]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2,3,4,5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,3,5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(param_grid)
        rf_Model = RandomForestRegressor()
        from sklearn.model_selection import GridSearchCV
        rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
        rf_Grid.fit(X_train, y_train)
        rf_Grid.best_params_
        best=rf_Grid.best_params_
        print(rf_Grid.best_params_)
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=best['n_estimators'], 
                                          bootstrap=best['bootstrap'], 
                                          max_depth=best['max_depth'], 
                                          max_features=best['max_features'], 
                                          min_samples_leaf=best['min_samples_leaf'], 
                                          min_samples_split=best['min_samples_split']) 
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_pred_1 = regressor.predict(X_train)
        y_pred=(y_pred*389.5548)+127.4741
        y_pred_1=(y_pred_1*389.5548)+127.4741
        y_test=(y_test*389.5548)+127.4741
        y_train=(y_train*389.5548)+127.4741
        from sklearn import metrics
        print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('Mean Absolute Error 1:',metrics.mean_absolute_error(y_train, y_pred_1))
        print('Mean Squared Error 1:',metrics.mean_squared_error(y_train, y_pred_1))
        print('Root Mean Squared Error 1:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_1)))
        print('r2:',metrics.r2_score(y_test, y_pred))
        print('r2:',metrics.r2_score(y_train, y_pred_1)) 
        df = pd.DataFrame(y_pred)
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred6.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred_16.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_test6.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_train6.csv")
    elif i==7:
        #Reading CSV datafiles
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala_17.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7306,:]
        Test = dataset.iloc[7306:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [50,100,200,300]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [5,10,30]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2,3,4,5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,3,5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(param_grid)
        rf_Model = RandomForestRegressor()
        from sklearn.model_selection import GridSearchCV
        rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
        rf_Grid.fit(X_train, y_train)
        rf_Grid.best_params_
        best=rf_Grid.best_params_
        print(rf_Grid.best_params_)
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=best['n_estimators'], 
                                          bootstrap=best['bootstrap'], 
                                          max_depth=best['max_depth'], 
                                          max_features=best['max_features'], 
                                          min_samples_leaf=best['min_samples_leaf'], 
                                          min_samples_split=best['min_samples_split']) 
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_pred_1 = regressor.predict(X_train)
        y_pred=(y_pred*389.5548)+127.4741
        y_pred_1=(y_pred_1*389.5548)+127.4741
        y_test=(y_test*389.5548)+127.4741
        y_train=(y_train*389.5548)+127.4741
        from sklearn import metrics
        print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('Mean Absolute Error 1:',metrics.mean_absolute_error(y_train, y_pred_1))
        print('Mean Squared Error 1:',metrics.mean_squared_error(y_train, y_pred_1))
        print('Root Mean Squared Error 1:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_1)))
        print('r2:',metrics.r2_score(y_test, y_pred))
        print('r2:',metrics.r2_score(y_train, y_pred_1)) 
        df = pd.DataFrame(y_pred)
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred7.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred_17.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_test7.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_train7.csv")
    elif i==8:
        #Reading CSV datafiles
        dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Excel Data\\ghala_18.csv')
        dataset = dataset.set_index(['Date'])
        Train = dataset.iloc[:7306,:]
        Test = dataset.iloc[7306:,:]
        X_train = Train.iloc[:,:-1]
        y_train = Train.iloc[:, -1]
        X_test = Test.iloc[:,:-1]
        y_test = Test.iloc[:, -1]
        #split data
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [50,100,200,300]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [5,10,30]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2,3,4,5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,3,5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(param_grid)
        rf_Model = RandomForestRegressor()
        from sklearn.model_selection import GridSearchCV
        rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
        rf_Grid.fit(X_train, y_train)
        rf_Grid.best_params_
        best=rf_Grid.best_params_
        print(rf_Grid.best_params_)
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=best['n_estimators'], 
                                          bootstrap=best['bootstrap'], 
                                          max_depth=best['max_depth'], 
                                          max_features=best['max_features'], 
                                          min_samples_leaf=best['min_samples_leaf'], 
                                          min_samples_split=best['min_samples_split']) 
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_pred_1 = regressor.predict(X_train)
        y_pred=(y_pred*389.5548)+127.4741
        y_pred_1=(y_pred_1*389.5548)+127.4741
        y_test=(y_test*389.5548)+127.4741
        y_train=(y_train*389.5548)+127.4741
        plt.rcParams['figure.figsize']=(12,8)
        x_ax=range(len(X_test))
        plt.plot(x_ax, y_test, label='observed', color='k', linestyle='-')
        plt.plot(x_ax, y_pred, label='predicted', color='g', linestyle='--')  
        from sklearn import metrics
        print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('Mean Absolute Error 1:',metrics.mean_absolute_error(y_train, y_pred_1))
        print('Mean Squared Error 1:',metrics.mean_squared_error(y_train, y_pred_1))
        print('Root Mean Squared Error 1:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_1)))
        print('r2:',metrics.r2_score(y_test, y_pred))
        print('r2:',metrics.r2_score(y_train, y_pred_1)) 
        df = pd.DataFrame(y_pred)
        df.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred8.csv")
        df1 = pd.DataFrame(y_pred_1)
        df1.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_pred_18.csv")
        y_test.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_test8.csv")
        y_train.to_csv("C:\\Users\\Lenovo\\Desktop\\Excel Data\\y_train8.csv")