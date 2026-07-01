import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

dataframe = pd.read_csv("Decision Tree Regression/housing_data.csv")
# print(dataframe.head())

# print(dataframe.isnull().values.any())
dataframe = dataframe.drop(columns=["id","date"])
# dataframe["date"] = pd.to_datetime(dataframe["date"])

dataframe["zipcode"] = pd.to_numeric(dataframe["zipcode"])

traget = "price"

numerical = ["sqft_living","sqft_lot","sqft_above","sqft_basement","yr_built","yr_renovated","sqft_living15","sqft_lot15"]
categorical = ["floors","waterfront","view","condition","grade"]

for colum in numerical:
    if colum != traget:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=dataframe[colum],y=dataframe[traget])
        plt.title(f"{colum} vs {traget}")
        plt.close()

for colum in categorical:
    if colum != traget:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=dataframe[colum],y=dataframe[traget])
        plt.title(f"{colum} vs {traget}")
        plt.close()

from sklearn.model_selection import train_test_split

X = dataframe.drop(columns = "price")
y = dataframe[traget]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


dt_model = DecisionTreeRegressor(max_depth=15,
    min_samples_split=20,
    min_samples_leaf=5,
    random_state=42
)
dt_model.fit(X_train,y_train)
y_predict = dt_model.predict(X_test)
'''
Dont use this for regression problems as we are calculating for PRICE it is a regression problem
print(confusion_matrix(y_predict,y_test))
print(classification_report(y_test,y_predict))
classification_report(...)
accuracy_score(...)
DecisionTreeClassifier

'''

mse = mean_squared_error(y_test,y_predict)
msrt = np.sqrt(mse)
mae = mean_absolute_error(y_test,y_predict)
r_score = r2_score(y_test,y_predict)


print(msrt)
print(mae)
print(r_score)

# feature_names = dataframe.drop(columns=["price"]).columns

# importance = pd.Series(
#     dt_model.feature_importances_,
#     index=feature_names
# ).sort_values(ascending=False)

importance = pd.Series(
    dt_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print(importance)

importance.plot(kind="bar", figsize=(10, 5))
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()