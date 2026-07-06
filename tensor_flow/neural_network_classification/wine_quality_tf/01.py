import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess data
df = pd.read_csv("tensor_flow/neural_network_classification/wine_quality_tf/wine-reviews.csv")

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "designation", "price", "region_1", "region_2", 
                       "taster_name", "taster_twitter_handle", "province", "country"], errors='ignore')

# Handle missing values
df["variety"] = df["variety"].fillna(df["variety"].mode()[0])
df["description"] = df["description"].fillna("")

# Create classification labels based on quality threshold (80th percentile)
target = "points"
quality_threshold = df[target].quantile(0.8)
df["quality_class"] = (df[target] > quality_threshold).astype(int)  # 1: High quality, 0: Low quality

# Prepare features and target
X = df[["description"]]
y = df["quality_class"].astype(np.int32)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts()}")
print(f"Quality threshold: {quality_threshold}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Load pre-trained embedding layer
embedding_url = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"
hub_layer = hub.KerasLayer(embedding_url, trainable=False, dtype=tf.string)

# Build text classification model
tf_model = keras.Sequential([
    keras.Input(name="description", dtype=tf.string, shape=()),
    hub_layer,
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")  # Binary classification
])

# Compile model
tf_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
)

# Print model summary
print("\nModel Summary:")
tf_model.summary()

# Train model with early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = tf_model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1,
    batch_size=64,
    callbacks=[early_stopping]
)

# Make predictions
y_pred_probs = tf_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\n" + "="*50)
print("PREDICTION RESULTS")
print("="*50)
print(f"\nPredicted probabilities (first 10): {y_pred_probs[:10].flatten()}")
print(f"\nPredicted classes (first 10): {y_pred[:10]}")

# Evaluation metrics
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality'])}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('tensor_flow/neural_network_classification/wine_quality_tf/training_history.png', dpi=300)
plt.show()

print("\n✓ Training history plot saved!")
