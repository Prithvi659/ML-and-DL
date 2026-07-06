import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

columns = ["flength","fwidth","fsize","fconc","fconc1","fasym","fm3long","fm3trans","falpha","fdist","class"]
dataframe = pd.read_csv('C:/prithvi/ml/data prep/magic04.data',names = columns)
# print(dataframe.head())

dataframe["class"] = (dataframe["class"] == "g").astype(int)
# print(dataframe.head())

for label in columns:
    plt.hist(dataframe[dataframe["class"] == 1][label],color= "red",label="gamma",density = True , alpha= 0.7)
    plt.hist(dataframe[dataframe["class"] == 0][label],color= "blue",alpha = 0.7,label="hedron",density=True)
    plt.title(label)
    plt.ylabel("probality")
    plt.xlabel(label)
    plt.legend()
    # plt.show()

train ,valid ,test = np.split(dataframe.sample(frac= 1),[int(0.6 * len(dataframe)),int(0.8 * len(dataframe))])

def split_feature_X_lables_y(df):
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    return X,y

# Apply the function to each dataset
X_train,y_train = split_feature_X_lables_y(train)
X_valid,y_valid = split_feature_X_lables_y(valid)
X_test,y_test = split_feature_X_lables_y(test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# normal tranform for validation and testing
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

ros = RandomOverSampler(random_state= 42)

X_train,y_train = ros.fit_resample(X_train,y_train)

# Why are you doing it? it is mainly for convenience.inspect it easily.

# But for training a model, you don't need this

train = np.hstack((X_train , np.reshape(y_train, (-1,1))))
valid = np.hstack((X_valid ,np.reshape(y_valid , (-1,1))))
test = np.hstack((X_test ,np.reshape(y_test , (-1,1))))


nb_model = GaussianNB()
nb_model.fit(X_train,y_train)
y_predict_2 = nb_model.predict(X_test)


print(classification_report(y_test,y_predict_2))