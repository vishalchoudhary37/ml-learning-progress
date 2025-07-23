import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("data.csv")

# Features (X) and Target (y)
X = data[['Experience']]
y = data['Salary']

# Split into training and testing sets
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model on training set
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
score = r2_score(y_test, y_pred)

print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Test data predictions:", y_pred)
print("R² Score on test data:", score)
