import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data_multiple.csv")
print("Data preview:\n", data.head())

# Independent variables (multiple)
X = data[['Experience', 'Projects']]
# Dependent variable
y = data['Salary']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Show coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict using the model
predictions = model.predict(X)
print("Predictions:\n", predictions)
