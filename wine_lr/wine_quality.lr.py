import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

'''   raise KeyError(key) from err
KeyError: 'quality' 
the issue is that your CSV is semicolon-separated (;), not comma-separated. use sep= '''

dataframe = pd.read_csv("linear_regression\winequality.csv",sep=";") 
# print(dataframe.head())

traget = "quality"
for colum in dataframe.columns:

    if colum != traget:
        plt.figure(figsize=(6,4))
        sb.scatterplot(x=dataframe[colum],y=dataframe[traget])
        sb.regplot(x=dataframe[colum],y=dataframe[traget])
        plt.title(f"{colum} vs {traget}")
        plt.xlabel(colum)
        plt.ylabel("quality")
        # plt.show()

train, validate, test = np.split(
    dataframe.sample(frac=1),
    [
        int(0.6 * len(dataframe)),
        int(0.8 * len(dataframe))
    ]
)

def models_target_split(df):
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    return X ,y

X_train,y_train = models_target_split(train)
X_valid,y_valid = models_target_split(validate)
X_test,y_test = models_target_split(test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Oversampling is not used for linear regression.

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)

# print(lr_model.score(X_test,y_test))
# print(lr_model.score(X_valid,y_valid))
# print(lr_model.score(X_test,y_test))

y_predict = lr_model.predict(X_test)
mean_squared_error(y_test,y_predict)
mean_absolute_error(y_test,y_predict)
r2_score(y_test,y_predict)  #r2_score(actual_values, predicted_values)
'''output values
0.3442637532050534  Lower is better
0.4501640958645476
0.3371077787100889

For the Wine Quality dataset, this is a normal result for Linear Regression
 because the relationship between chemical properties and quality is not purely linear.'''

cofficient = lr_model.coef_
''' Coefficients
[-0.0348 -0.1971 -0.0131 -0.0412 -0.0461  0.0828
 -0.1389  0.0246 -0.0915  0.1320  0.2949]

These are your model's weights (lr_model.coef_).

Your regression equation is:
Each number tells how much that feature affects the prediction.
'''


intersept = lr_model.intercept_
'''
5.636079249217934
This is your model's predicted wine quality for one sample.
'''

X = np.linspace(0,20,100)
y = np.dot(X_test[:100], cofficient) + intersept
'''
y = cofficient * X + intersept (ValueError: operands could not be broadcast together with shapes (11,) (100,))
coefficient → (11,)
X            → (100,)
Python cannot multiply them because the dimensions do not match.

finally use   y = np.dot(X_test[:100], cofficient) + intersept
This gives 100 predictions.
'''

plt.scatter(X,y)
plt.xlabel("features")
plt.ylabel("quality")
plt.title("wine quality")
plt.grid()
plt.show()