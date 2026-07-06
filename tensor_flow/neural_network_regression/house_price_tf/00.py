import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

dataframe = pd.read_csv("tensor_flow/neural_network_regression/house_price_tf/Bengaluru_House_Data.csv")


def convert_sqft(x):
    try:
        return float(str(x).split()[0])
    except:
        return np.nan

dataframe["total_sqft"] = dataframe["total_sqft"].apply(convert_sqft)
dataframe = dataframe.drop(columns=["society"])

# Numeric features → Fill with mean or median.
# Categorical features → Fill with mode
dataframe["location"] = dataframe["location"].fillna(dataframe["location"].mode()[0])
dataframe["size"] = dataframe["size"].fillna(dataframe["size"].mode()[0])
dataframe["bath"] = dataframe["bath"].fillna(dataframe["bath"].mean())
dataframe["balcony"] = dataframe["balcony"].fillna(dataframe["balcony"].mean())


# print(dataframe.isnull().sum())
traget = "price"
numerical = ["total_sqft","bath","balcony"]
categorical = ["area_type","availability","location","size"]

for col in numerical:
    if col != traget:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=dataframe[col],y=dataframe[traget])
        plt.xlabel(col)
        plt.ylabel(dataframe[traget])
        plt.title("impact of the model")
        plt.close()

for col in categorical:
    if col != traget:
        plt.figure(figsize=(10,5))
        sns.boxplot(x=dataframe[col],y=dataframe[traget])
        plt.xlabel(col)
        plt.ylabel(dataframe[traget])
        plt.title("impact of the model")
        plt.close()

dataframe = dataframe.dropna()

X = dataframe.drop(columns=[traget])
X = pd.get_dummies(X, drop_first=True)  # Now all text becomes 0/1 numbers.
y = dataframe[traget]


# create model
X_train ,X_test , y_train , y_test = train_test_split(X,y,random_state= 42,test_size=0.2)

x_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# create a TF Model
tf_model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Compile the model
tf_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = tf.keras.losses.Huber(),metrics=[tf.keras.metrics.RootMeanSquaredError()])


checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.keras",
    monitor="val_loss",save_best_only=True
)
csv_logger = tf.keras.callbacks.CSVLogger("traing_log.csv")

# fit the data
# epochs = one complete pass through dataset of training
# 50 epochs = check 50 times training data
# verbose = how much training progress is printed 0 = no print ,1 = more print,2 = less print

history = tf_model.fit(X_train,y_train_scaled,epochs = 100,
     validation_split = 0.2,verbose = 2,callbacks=[checkpoint,csv_logger])

predict_scaled = tf_model.predict(X_test)
y_predict = y_scaler.inverse_transform(predict_scaled).flatten()

mse = mean_squared_error(y_test,y_predict)
mae = mean_absolute_error(y_test,y_predict)
r_score = r2_score(y_test,y_predict)

print(f"MSE : {mse} \nMAE : {mae} \nR SCORE : {r_score}")

# Save predictions to CSV
results_df = pd.DataFrame({'actual': y_test.values,'predicted': y_predict})
results_df.to_csv('predictions_results.csv', index=False)
print("Predictions saved to 'predictions_results.csv'")

# Save training history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)
print("Training history saved to 'training_history.csv'")

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test,y=y_predict,alpha=0.5)
plt.plot([y_test.min(),y_test.max()],          # This is not plotting your prediction
         [y_test.min(),y_test.max()],color = "red")  # It's drawing the reference line y = x, which represents perfect predictions.
plt.title("actual vs predicated")
plt.xlabel("actual")
plt.ylabel("predicted")
plt.savefig("actual_vs_predicted.png", dpi=100, bbox_inches='tight')
# plt.show()  63

tf_model.save("house_price_tf.h5")