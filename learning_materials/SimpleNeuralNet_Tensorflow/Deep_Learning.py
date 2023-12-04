# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:29:54 2023

@author: Jianhua Mei
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#%% Load MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#%% Scale to [0,1]
train_images = train_images / 255
test_images = test_images / 255

#%% reshape Flatten
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

#%% Define Model
def build_model(input_dim, output_dim, nr_neurons):
    model = keras.Sequential([
        layers.Dense(nr_neurons, activation='sigmoid', input_shape=[input_dim]),
        layers.Dense(nr_neurons, activation='sigmoid'),
        layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])
    return model

#%% Parameter
input_dim = 784
output_dim = 10
EPOCHS = 1000
neurons = 128

#%% Train Model
model = build_model(input_dim, output_dim, neurons)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
time_start = time.time()
history = model.fit(train_images, train_labels, epochs=EPOCHS,
                    validation_split=0.1, batch_size = 10, callbacks=[early_stop])
time_end = time.time()
print('Model Training Time Cost:', time_end - time_start, 's')

#%% Training Loss and validation Loss
training_loss = history.history["accuracy"]
val_loss = history.history["val_accuracy"]
epoch_count = range(1, len(training_loss) + 1)

plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, val_loss, "b-")
plt.legend(["Training accuracy", "Validation accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST Multi-Layer Perceptron Training Progress")
plt.show()

#%% Out of sample Test
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy}")
test_predictions = model.predict(test_images).argmax(axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('MNIST Multi-Layer Perceptron Confusion Matrix')
plt.show()
