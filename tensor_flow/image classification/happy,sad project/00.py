import matplotlib.pyplot as plt
from pathlib import Path
import filetype
import tensorflow as tf
import numpy as np
import cv2


data_dir = Path(__file__).parent / "data"
allowed_image_type = {"jpg", "jpeg", "bmp", "png"}

for class_dir in data_dir.iterdir(): #Loop through every folder inside data (e.g., happy, sad)
    if not class_dir.is_dir():
        continue                 #Skip anything that isn't a folder.

    for image_path in class_dir.iterdir(): #Loop through every file in that folder.
        try:
            kind = filetype.guess(image_path)
            if kind is None or kind.extension not in allowed_image_type:
                image_path.unlink()
        except Exception:
            pass

dataset = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(256, 256), batch_size=32)
dataset = dataset.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

print(len(dataset))

train_size = int(len(dataset) * 0.7)
val_size = int(len(dataset) * 0.2)
test_size = len(dataset) - train_size - val_size


train = dataset.take(train_size)
val = dataset.skip(train_size).take(val_size)
test = dataset.skip(train_size + val_size)

tf_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),1,activation="relu",
        input_shape = (256,256,3)),
        tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32,(3,3),1,activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#compile model
tf_model.compile(tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = tf.keras.losses.BinaryCrossentropy(), metrics = ["accuracy"])

# train model
history = tf_model.fit(train,validation_data =val,epochs = 50)

plt.plot(history.history.get("accuracy", []), label="accuracy")
plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
plt.legend()
plt.show()

plt.plot(history.history.get("loss", []), label="loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.legend()
plt.show()


for images, labels in test.take(1):
    print(labels.numpy())
    print(images.shape)

tf_model = tf.keras.models.save("happy sad model.h5")