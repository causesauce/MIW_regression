import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = np.loadtxt('./dane4.txt')

X = data[:, 0].reshape((-1, 1))
y = data[:, 1].reshape((-1, 1))


plt.scatter(X, y)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

print('parameters(linear): ', lin_reg.coef_, lin_reg.intercept_)

print('Mean squared error(linear): ', mean_squared_error(y_test, y_pred))

plt.title('linear regression')
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

# nonlinear regression
poly = PolynomialFeatures(degree=5, include_bias=False)

poly_features = poly.fit_transform(X.reshape((-1, 1)))

poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)
y_pred2 = poly_reg_model.predict(poly_features)

print('Mean squared error(nonlinear): ', mean_squared_error(y, y_pred2))

plt.title("nonlinear regression")
plt.scatter(X, y, color='black')
plt.plot(X, y_pred2, color='blue', linewidth=3)
plt.show()
