import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
data = pd.read_csv("data_poly.csv")

X = data[['Level']]
y = data['Salary']

# Transform features to polynomial features (degree=2 or 4)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# Predict for visualization
X_grid = np.arange(min(X['Level']), max(X['Level'])+0.1, 0.1).reshape(-1,1)
y_grid_pred = model.predict(poly.transform(X_grid))

# Print coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X_grid, y_grid_pred, color='red')
plt.title("Polynomial Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
