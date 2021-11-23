# Christopher Gjersing (NSHE: 5004590677)
# Section 1002 | Fall 2021
# Assignment #3: Linear Regression Algorithms

#Predicting MPG of vehicles given displacement (DP), horsepower (HP), weight (WT), and acceleration (AC)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, scale


#--- DATA PROCESSING ---
# Read data, convert into X and y numpy arrays
df = pd.read_fwf('auto-mpg.data')
y = df['MPG'].to_numpy()
df.drop(df.columns[[0, 1, 6, 7, 8]], axis=1, inplace=True)
#df['Bias'] = 1 # comment out if using fit_intercept = True
X = df.to_numpy()

#--- OLS ALGORITHM ---
# Initialize model, split data, fit model, run prediction
ols = LinearRegression(fit_intercept=True) # Use False if using the Bias parameter vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
ols.fit(X_train, y_train)
olsTrainPredict = ols.predict(X_train)
prediction = ols.predict(X_test)

# ** OLS Error Plotting code: Remove to see prediction vs actual w/ error lines
# index = np.arange(79)
# fig = plt.figure()
# for i in range(len(index)):
#     plt.vlines(i, prediction[i], y_test[i], colors='black', alpha = 0.5)
# plt.scatter(index, prediction, color = 'black', label = 'Predicted MPG', zorder = 2)
# plt.scatter(index, y_test, color = 'red', label = 'Actual MPG', zorder = 2)
# plt.legend()
# fig.suptitle('OLS Error', fontsize = 20)
# plt.show()

#Print bias, solution, and evalution metrics of OLS
print('====== OLS RESULTS ======')
print('  Bias (Intercept): ', ols.intercept_)
print('   Coefficents (w): ', ols.coef_)
print('      Training MAE: ', mean_absolute_error(y_train, olsTrainPredict))
print('      Training MSE: ', mean_squared_error(y_train, olsTrainPredict))
print('       Training R2: ', r2_score(y_train, olsTrainPredict))
print('          Test MAE: ', mean_absolute_error(y_test, prediction))
print('          Test MSE: ', mean_squared_error(y_test, prediction))
print('           Test R2: ', r2_score(y_test, prediction), '\n')

#--- Gradient Descent (GD) ALGORITHM ---
#Can adjust loss function, learning rate schedule, and learning rate 'eta0' value
gd = SGDRegressor(loss='squared_error', eta0=0.01, n_iter_no_change=400, max_iter=5000)

#Scaling data to remove the mean and scale to unit variance, then splitting into 80/20 train/test
scaler = StandardScaler()
gd_X = scaler.fit_transform(X)
gd_X_train, gd_X_test, gd_y_train, gd_y_test = train_test_split(gd_X, y, test_size=0.2)

gd.fit(gd_X_train, gd_y_train)
gdTrainPredict = gd.predict(gd_X_train)
gdTestPredict = gd.predict(gd_X_test)

#Print bias, solution, and evaluation metrics of GD
print('====== GD RESULTS =======')
print('     Learning Rate: ', gd.eta0)
print('    Iterations (n): ', gd.n_iter_)
print('  Bias (Intercept): ', gd.intercept_)
print('   Coefficents (w): ', gd.coef_)
print('      Training MAE: ', mean_absolute_error(gd_y_train, gdTrainPredict))
print('      Training MSE: ', mean_squared_error(gd_y_train, gdTrainPredict))
print('       Training R2: ', r2_score(gd_y_train, gdTrainPredict))
print('          Test MAE: ', mean_absolute_error(gd_y_test, gdTestPredict))
print('          Test MSE: ', mean_squared_error(gd_y_test, gdTestPredict))
print('           Test R2: ', r2_score(gd_y_test, gdTestPredict), '\n')