
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  # features
y = np.array([3, 4, 2, 5, 6, 7])                 # target values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Slope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
