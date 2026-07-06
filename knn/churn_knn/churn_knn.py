import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report

data = pd.read_csv(r'C:\prithvi\ml\knn\1_Churn_Modelling.csv')
''' Adding r prefix to make it a raw string (treats backslashes literally)
    Removing the extra space in the filename'''
# print(data.head)
# print(data.tail)
# print(data.info)

# starting knn alogrithm

X = data.loc[:,["CreditScore","Age","Tenure","Balance","NumOfProducts",
                "HasCrCard","IsActiveMember","EstimatedSalary"]]

Y = data.Exited

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)

model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train,Y_train)

Y_predict = model.predict(X_test)

# evalute y_predict
accuracy_score = (Y_predict,Y_test)
print(classification_report(Y_predict,Y_test))