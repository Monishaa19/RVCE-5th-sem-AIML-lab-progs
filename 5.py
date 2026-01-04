import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris=datasets.load_iris()
x = iris.data[:, :2]
y=(iris.target!=0)*1

sc=StandardScaler()
x=sc.fit_transform(x)
print(x[:5])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        return -np.mean(y*np.log(h) + (1-y)*np.log(1-h))

    def fit(self, X, y, num_iterations=1000, learning_rate=0.01):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(num_iterations):
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)

            dw = (1/n_samples) * np.dot(X.T, (h - y))
            db = (1/n_samples) * np.sum(h - y)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(z) >= 0.5).astype(int)

    
lr=LogisticRegression()
lr.fit(x_train,y_train,learning_rate=0.01,num_iterations=200)

y_pred=lr.predict(x_test)
np.mean(y_pred==y_test)
print(np.mean(y_pred==y_test))

import matplotlib.pyplot as plt
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)

plt.title('Logistic Regression Decision Boundaries')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
