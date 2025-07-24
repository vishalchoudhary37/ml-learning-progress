import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("data_poly.csv")
X = data[['Level']].values
y = data['Salary'].values

# Create and train the model
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Predict a specific level
level_to_predict = 6.5
predicted_salary = regressor.predict([[level_to_predict]])
print(f"Predicted salary for level {level_to_predict}: {predicted_salary[0]}")

# Plot high-resolution curve
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X_grid, regressor.predict(X_grid), color='orange', label='Random Forest model')
plt.title('Random Forest Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
