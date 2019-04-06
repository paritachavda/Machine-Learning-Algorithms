import pandas as pd
import numpy as np

def feature_normalize(X):
    n_features = X.shape[1]
    means = np.array([np.mean(X.values[:,i]) for i in range(n_features)])
    stddevs = np.array([np.std(X.values[:,i]) for i in range(n_features)])
    normalized = (X - means) / stddevs

    return normalized
def compute_cost(X, y, theta):
    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))

def gradient_descent_multi(X, y, theta, alpha, iterations):
    m = len(X)

    for i in range(iterations):
        gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - y)
        theta = theta - alpha * gradient

    return theta

data = pd.read_csv("boston_housing.csv")
x = data.drop('MEDV',axis=1)
y = data.MEDV 

X=feature_normalize(x)
X.insert(0,'x0',1)
theta = np.zeros(X.shape[1])
Y = (y - y.mean())/y.std()
cost= compute_cost(X,Y,theta)


alpha = 0.025
iterations=1000
theta = gradient_descent_multi(X, Y, theta, alpha, iterations)
cost = compute_cost(X, Y, theta)

print('Weight vector:', theta)
print('error:', cost)

print('Predicted values :')

x1 = [1.0,0.085, 13.0, 10.5, 1.0, 0.8, 4.78, 39.0, 5.5, 5.5, 331.0, 13.3, 390.5, 17.71]
X1 = (x1 - np.mean(x1))/np.std(x1) 
predict = np.matmul(theta,X1)

final_value = (predict*y.std())+ y.mean()
print(x1)
print(final_value)

x2 = [1.0,0.17899, 0, 9.69, 0, 0.585, 5.67, 28.8, 2.7986, 6, 391, 19.2, 393.29, 17.6]

X2 = (x2 - np.mean(x2))/np.std(x2) 
predict2 = np.matmul(theta,X2)

final_value2 = (predict2*y.std())+ y.mean()
print(x2)
print(final_value2)
