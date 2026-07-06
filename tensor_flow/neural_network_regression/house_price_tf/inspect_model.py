import tensorflow as tf
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved model (use the .keras file)
model_path = os.path.join(script_dir, 'house_price_tf.h5')
model = tf.keras.models.load_model(model_path)
model.summary()

