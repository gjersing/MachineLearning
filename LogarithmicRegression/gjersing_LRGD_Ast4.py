# Christopher Gjersing | NSHE: 5004590677
# Assignment 4 | CS 422 1002 | 2021 Fall 
# Log Regression w/ GD : Banknote authenticity

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss

#Split into X, y and add bias parameter vector
data = pd.read_csv('data_banknote_authentication.txt', header=None)
X = data.copy()
X[4] = 1
y = data[4].to_numpy()

#Split and train, then predict for both train and test and calculate confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
logRegression = SGDClassifier(loss='log', fit_intercept=False, learning_rate='constant', eta0=0.01)
logRegression.fit(X_train, y_train)
train_prediction = logRegression.predict(X_train)
train_matrix = confusion_matrix(y_train, train_prediction)
test_prediction = logRegression.predict(X_test)
test_matrix = confusion_matrix(y_test, test_prediction)

#Everything past here is just results print statements
print('==== Logistic Regression with Gradient Descent Results ====\n')
print('Solution (w): ', logRegression.coef_)
print('Convergence Iterations: ', logRegression.n_iter_)
print('Learning Rate (eta0): ', logRegression.eta0, '\n')

print('---- Train Evaluation Metrics ----\n', '   Matrix\n', train_matrix)
n_train = len(y_train)
tn, fp, fn, tp = train_matrix.ravel()
sens = tp/(tp+fn)
prec = tp/(tp+fp)
print('   Accuracy: ', (tn + tp)/n_train)
print('Sensitivity: ', sens)
print('Specificity: ', tn/(tn+fp))
print('   F1 Score: ', (2*sens*prec)/(sens+prec))
print('   Log Loss: ', log_loss(y_train, train_prediction, normalize=True), '\n')

print('---- Test Evaluation Metrics ----\n', '   Matrix\n', test_matrix)
n_test = len(y_test)
tn, fp, fn, tp = test_matrix.ravel()
sens = tp/(tp+fn)
prec = tp/(tp+fp)
print('   Accuracy: ', (tn + tp)/n_test)
print('Sensitivity: ', sens)
print('Specificity: ', tn/(tn+fp))
print('   F1 Score: ', (2*sens*prec)/(sens+prec))
print('   Log Loss: ', log_loss(y_test, test_prediction, normalize=True))
print('===========================================================\n')