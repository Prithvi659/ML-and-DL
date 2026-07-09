import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

path = "C:/prithvi/ml/linear_regression/salary_data.csv"

dataframe = pd.read_csv(path)


plt.figure(figsize=(6,4))
sea.scatterplot(x=dataframe["YearsExperience"],y=dataframe["Salary"])
plt.title("salary data")
plt.xlabel(["YearsExperience"])
plt.ylabel(["Salary"])
# plt.show()

train , valid , test = np.split(dataframe.sample(frac= 1),[int (0.6 * len(dataframe)),int (0.8 * len(dataframe))])

def model_traget(df):
    X = df[df.columns[:-1]].values
    y = df[df.columns[1]].values

    return X,y

X_train , y_train = model_traget(train)
X_valid , y_valid = model_traget(valid)
X_test , y_test = model_traget(test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

lr_model = LinearRegression()

lr_model.fit(X_train,y_train)
y_predict = lr_model.predict(X_test)

mse = mean_squared_error(y_test,y_predict)
mae =mean_absolute_error(y_test,y_predict)
r2 =r2_score(y_test,y_predict)

cofficient = lr_model.coef_
intercept = lr_model.intercept_


plt.figure(figsize=(6,4))
plt.scatter(y_test, y_predict, color="blue", label="Predictions")
min_val, max_val = min(y_test.min(), y_predict.min()), max(y_test.max(), y_predict.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect prediction")
plt.title("salary data")
plt.xlabel("actual")
plt.ylabel("predicted")
plt.grid()
plt.show()

print("mean absoult err: ",mae)
print(f"RMSE: {np.sqrt(mse):.2f}")
print("mean square err: ",mse)
print("r ** 2 score: ",mae)
