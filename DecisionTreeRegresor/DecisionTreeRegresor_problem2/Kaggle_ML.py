import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
melbourne_data = pd.read_csv('melb_data.csv')
filtered_melbourne_data = melbourne_data.dropna(axis=0)
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]
print(X.describe)
print(X.head())
melbourne_model = DecisionTreeRegressor(random_state=2)
melbourne_model.fit(X, y)
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
print(mean_absolute_error)