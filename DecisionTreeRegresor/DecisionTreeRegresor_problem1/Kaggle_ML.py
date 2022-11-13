import pandas as pd
from sklearn.tree import DecisionTreeRegressor
home_data = pd.read_csv('iowa_house_price_test_data.csv')
print(home_data.columns)
y = home_data.SalePrice
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[feature_names]
print(X.describe)
print(X.head())
iowa_model = DecisionTreeRegressor(random_state=2)
iowa_model.fit(X,y)
predictions = iowa_model.predict(X)
print(predictions)