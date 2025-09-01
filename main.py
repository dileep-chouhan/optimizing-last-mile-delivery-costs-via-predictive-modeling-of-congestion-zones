import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# --- 1. Synthetic Data Generation ---
# Generate synthetic data for delivery times and congestion levels
np.random.seed(42)  # for reproducibility
num_deliveries = 500
data = {
    'Delivery_Time': np.random.normal(loc=30, scale=5, size=num_deliveries), # Delivery time in minutes
    'Distance': np.random.uniform(5, 20, size=num_deliveries), # Distance in km
    'Time_of_Day': np.random.choice(['Morning', 'Afternoon', 'Evening'], size=num_deliveries),
    'Congestion_Level': np.random.randint(1, 5, size=num_deliveries) # Congestion level (1-low, 4-high)
}
df = pd.DataFrame(data)
# Add some noise to simulate real-world scenarios
df['Delivery_Time'] += np.random.normal(loc=0, scale=2, size=num_deliveries)
df['Delivery_Time'] = np.maximum(df['Delivery_Time'], 0) # Ensure no negative delivery times
# --- 2. Data Cleaning and Feature Engineering ---
# One-hot encode the categorical feature 'Time_of_Day'
df = pd.get_dummies(df, columns=['Time_of_Day'], drop_first=True)
# --- 3. Predictive Modeling ---
# Define features (X) and target (y)
X = df[['Distance', 'Congestion_Level', 'Time_of_Day_Afternoon', 'Time_of_Day_Evening']]
y = df['Delivery_Time']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# --- 4. Model Evaluation (Illustrative -  More robust evaluation needed for real-world application)---
#  In a real-world scenario, more rigorous model evaluation metrics would be used.
print(f"Model R-squared: {model.score(X_test, y_test)}")
# --- 5. Visualization ---
# Visualize the relationship between congestion level and delivery time
plt.figure(figsize=(8, 6))
plt.scatter(df['Congestion_Level'], df['Delivery_Time'])
plt.xlabel('Congestion Level')
plt.ylabel('Delivery Time (minutes)')
plt.title('Congestion Level vs. Delivery Time')
plt.savefig('congestion_vs_delivery_time.png')
print("Plot saved to congestion_vs_delivery_time.png")
#Visualize model predictions vs actual delivery times (Illustrative)
plt.figure(figsize=(8,6))
plt.scatter(y_test, model.predict(X_test))
plt.xlabel("Actual Delivery Time")
plt.ylabel("Predicted Delivery Time")
plt.title("Actual vs Predicted Delivery Times")
plt.savefig("actual_vs_predicted.png")
print("Plot saved to actual_vs_predicted.png")