# Christopher Gjersing (NSHE: 5004590677)
# Section 1002 | Fall 2021
# Assignment #5: MLP Regressor predicts MPG

# Using this to force sklearn to not give warnings for hitting max iterations
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Cut data into X matrix and Y vector
data = pd.read_csv('trim-auto-mpg')
y = data['MPG'].to_numpy()
data.drop(data.columns[[4]], axis=1, inplace=True)
X = data

#Split data and scale inputs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Initialize models and fit
mlp1 = MLPRegressor(hidden_layer_sizes=(100), activation='tanh', solver='sgd', learning_rate='constant', learning_rate_init=0.01, max_iter=2000).fit(X_train, y_train)
mlp2 = MLPRegressor(hidden_layer_sizes=(80,40), activation='identity', solver='sgd', learning_rate='invscaling', max_iter=2000).fit(X_train, y_train)
mlp3 = MLPRegressor(hidden_layer_sizes=(60,40,20,10), activation='relu', solver='lbfgs', max_iter=2000).fit(X_train, y_train)

#Run predictions and print metrics for mlp1
mlp1TrainPredict = mlp1.predict(X_train)
mlp1TrainMAE= mean_absolute_error(y_train, mlp1TrainPredict)
mlp1TrainMSE= mean_squared_error(y_train, mlp1TrainPredict)
mlp1TrainR2 = r2_score(y_train, mlp1TrainPredict)

mlp1TestPredict = mlp1.predict(X_test)
mlp1TestMAE= mean_absolute_error(y_test, mlp1TestPredict)
mlp1TestMSE= mean_squared_error(y_test, mlp1TestPredict)
mlp1TestR2 = r2_score(y_test, mlp1TestPredict)

print('MLP1 Train Metrics (MAE|MSE|R2): ', mlp1TrainMAE, ' | ', mlp1TrainMSE, ' | ', mlp1TrainR2)
print('MLP1 Test Metrics  (MAE|MSE|R2): ', mlp1TestMAE, ' | ', mlp1TestMSE, ' | ', mlp1TestR2)
print('MLP1 Iterations: ', mlp1.n_iter_)
print('------------------------------------------------------------------------------------------------------')

#Run predictions and print metrics for mlp2
mlp2TrainPredict = mlp2.predict(X_train)
mlp2TrainMAE = mean_absolute_error(y_train, mlp2TrainPredict)
mlp2TrainMSE = mean_squared_error(y_train, mlp2TrainPredict)
mlp2TrainR2 = r2_score(y_train, mlp2TrainPredict)

mlp2TestPredict = mlp2.predict(X_test)
mlp2TestMAE= mean_absolute_error(y_test, mlp2TestPredict)
mlp2TestMSE= mean_squared_error(y_test, mlp2TestPredict)
mlp2TestR2 = r2_score(y_test, mlp2TestPredict)

print('MLP2 Train Metrics (MAE|MSE|R2): ', mlp2TrainMAE, ' | ', mlp2TrainMSE, ' | ', mlp2TrainR2)
print('MLP2 Test Metrics  (MAE|MSE|R2): ', mlp2TestMAE, ' | ', mlp2TestMSE, ' | ', mlp2TestR2)
print('MLP2 Iterations: ', mlp2.n_iter_)
print('------------------------------------------------------------------------------------------------------')

#Run predictions and print metrics for mlp3
mlp3TrainPredict = mlp3.predict(X_train)
mlp3TrainMAE= mean_absolute_error(y_train, mlp3TrainPredict)
mlp3TrainMSE= mean_squared_error(y_train, mlp3TrainPredict)
mlp3TrainR2 = r2_score(y_train, mlp3TrainPredict)

mlp3TestPredict = mlp3.predict(X_test)
mlp3TestMAE= mean_absolute_error(y_test, mlp3TestPredict)
mlp3TestMSE= mean_squared_error(y_test, mlp3TestPredict)
mlp3TestR2 = r2_score(y_test, mlp3TestPredict)

print('MLP3 Train Metrics (MAE|MSE|R2): ', mlp3TrainMAE, ' | ', mlp3TrainMSE, ' | ', mlp3TrainR2)
print('MLP3 Test Metrics  (MAE|MSE|R2): ', mlp3TestMAE, ' | ', mlp3TestMSE, ' | ', mlp3TestR2)
print('MLP3 Iterations: ', mlp3.n_iter_)