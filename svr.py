import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data_poly.csv")
X = data[['Level']].values
y = data['Salary'].values

# Reshape y to 2D for scaling
y = y.reshape(-1, 1)

# Feature Scaling (very important for SVR)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y).ravel()

# Train SVR with RBF kernel
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X_scaled, y_scaled)

# Predict for a certain level
level_to_predict = 6.5
pred_scaled = svr_regressor.predict(sc_X.transform([[level_to_predict]]))
pred_original = sc_y.inverse_transform(pred_scaled.reshape(-1, 1))
print(f"Predicted salary for level {level_to_predict}: {pred_original[0][0]}")

# Plot results
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, sc_y.inverse_transform(svr_regressor.predict(X_scaled).reshape(-1,1)), color='red', label='SVR model')
plt.title('Support Vector Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
