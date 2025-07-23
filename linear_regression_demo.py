# Day 7: Linear Regression Demo
# Vishal's first ML project 🚀

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Create a simple dataset
# Suppose we want to learn the relationship between X and Y
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  # features
y = np.array([3, 4, 2, 5, 6, 7])                 # target values

# Step 2: Create and train the model
model = LinearRegression()
model.fit(X, y)

# Step 3: Make predictions
y_pred = model.predict(X)

# Step 4: Print model details
print("Slope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)

# Step 5: Plot
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
