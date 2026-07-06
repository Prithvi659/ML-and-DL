import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras

from sklearn.model_selection import train_test_split

df = pd.read_csv("tensor_flow/neural_network_classification/wine_quality_tf/wine-reviews.csv")

df=df.drop(columns=["Unnamed: 0","designation","price","region_1","region_2","taster_name","taster_twitter_handle","province","country"])
df["variety"] = df["variety"].fillna(df["variety"].mode()[0])
df["description"] = df["description"].fillna("")

target = "points"
X = df[["description"]]
y = df[target].astype(np.float32)

X_train , X_test ,y_train , y_test = train_test_split(X,y,random_state=42,test_size=0.2)

# this is used to covert big sentence into tensors
embedding = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"
hub_layer = hub.KerasLayer(embedding,trainable=True,dtype = tf.string)

# create tf model
tf_model = keras.Sequential()
tf_model.add(keras.Input(name = "description", dtype = tf.string,shape =()))
tf_model.add(hub_layer)
tf_model.add(keras.layers.Dense(128,activation="relu"))
tf_model.add(keras.layers.Dropout(0.2))
tf_model.add(keras.layers.Dense(64,activation="relu"))
tf_model.add(keras.layers.Dropout(0.2))
tf_model.add(keras.layers.Dense(32,activation="relu"))
tf_model.add(keras.layers.Dropout(0.2))
tf_model.add(keras.layers.Dense(1,activation="sigmoid")) # for classification = sigmoid and dense = 1

#compile model  when to use = BinaryCrossentropy() = Two classes (Yes/No, 0/1, Spam/Not Spam)
tf_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    loss = keras.losses.BinaryCrossentropy(),metrics = ["accuracy"])

# train model when data is big add batch_size
history = tf_model.fit(X_train,y_train,epochs= 50,validation_split = 0.2,verbose = 0,batch_size = 64)
y_predict = tf_model.predict(X_test)

print(history,y_predict)