import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Load dataset
data = pd.read_csv("data.csv")  # CSV should be in the same folder
print("Data preview:\n", data.head())

# Step 2: Prepare X and y
X = data[['Experience']]  # independent variable
y = data['Salary']        # dependent variable

# Step 3: Create and train the model
model = LinearRegression()
model.fit(X, y)

# Step 4: Check the model parameters
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 5: Predict salaries
predictions = model.predict(X)
print("Predictions:\n", predictions)

# Step 6: Plot results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, predictions, color='red', label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary (Linear Regression)')
plt.legend()
plt.show()
