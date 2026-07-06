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
# print(dataframe.head())

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

# this is maual prediction we can just use === y_predict = lr_model.predict(X_test) both are same
# X = np.linspace(0,20,100)
# y = np.dot(X_test, cofficient) + intercept
# print(len(X),len(y))


plt.figure(figsize=(6,4))
plt.scatter(y_test, y_predict, color="blue", label="Predictions")
min_val, max_val = min(y_test.min(), y_predict.min()), max(y_test.max(), y_predict.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect prediction")
'''Goal: draw a diagonal reference line showing "if every prediction were perfect, 
  this is where the points would sit.
" Then you can visually compare your actual orange/blue dots against that line.'''
plt.title("salary data")
plt.xlabel("actual")
plt.ylabel("predicted")
plt.grid()
plt.show()

print("mean absoult err: ",mae)


'''MAE — "On average, how far off was I?"
Imagine your model guesses someone's salary, and you compare it to their real salary. Do that for every person in your test set, ignore whether the guess was too high or too low (just look at the size of the miss), then average all those misses together. That's MAE.
If MAE = 5000, it means: "On a typical guess, I'm off by about 5000." Simple as that.
'''


print(f"RMSE: {np.sqrt(mse):.2f}")


'''
RMSE — "Same idea, but I really don't like being badly wrong"
If MAE and RMSE are close to each other → your model is consistently a little bit wrong, no major disasters.
If RMSE is much bigger than MAE → most guesses are fine, but a few are way off, and those few are dragging RMSE up.
Worth checking which data points those are.
'''


print("mean square err: ",mse)
print("r ** 2 score: ",mae)



'''
R² — "How much of the story does my model actually explain?"
Picture salary as something that should go up as experience goes up — that's the pattern.
R² tells you what fraction of that pattern your model successfully captured, on a scale from 0 to 1 (think of it like a percentage).

R² = 1 → your model nailed the relationship perfectly, every point falls exactly where predicted.
R² = 0 → your model is no better than just guessing "average salary" for everyone, ignoring experience entirely.
R² close to 0.9–1.0 (90–100%) → great, since salary-vs-experience is normally a clean, strong relationship,
 you should expect to land here.
R² low, or even negative → something's wrong — bad data split, bug in the code,
 or the relationship genuinely isn't linear for this dataset.
'''