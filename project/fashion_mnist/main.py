import tensorflow as tf
import numpy as np
from fashion_mnist.callbacks.get_callbacks import get_callbacks
from fashion_mnist.dataset.data_api import get_train_data
from fashion_mnist.dataset.data_api import get_validation_data
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import cv2
tf.enable_eager_execution()

def show(images, labels, name):
    grid_image = tf.concat([tf.concat([images[i*10 + j] for j in range(10)], 1) for i in range(len(images)//10)], 0)
    grid_image = tf.squeeze(grid_image)
    cv2.imwrite( name+"_sample.png", grid_image.numpy()*255)
            

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = np.expand_dims(train_images, axis = -1)/255
test_images = np.expand_dims(test_images, axis = -1)/255

train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size = 0.15, random_state = 42, shuffle = True)

training_data = get_train_data(train_images, train_labels)
feature_batch = training_data.take(1)
for images, labels in feature_batch:
    show(images, labels, 'train')

validation_data = get_validation_data(validation_images, validation_labels)
feature_batch = validation_data.take(1)
for images, labels in feature_batch:
    show(images, labels, 'validation')

inputLayer = tf.keras.layers.Input(shape = (28, 28, 1))
x = inputLayer

for i in range(2):
    x = tf.keras.layers.Conv2D(filters = 128*2**i, kernel_size = 3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(rate = 0.2, seed = 42)(x)
x = tf.keras.layers.Dense(512, activation = 'relu')(x)
output = tf.keras.layers.Dense(10, activation = 'softmax')(x)

model = tf.keras.Model(inputs = inputLayer, outputs = output)
model.compile('adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_crossentropy', 'accuracy'])


model.fit(x=training_data, validation_data=validation_data, epochs=100, verbose=1, callbacks=get_callbacks(), steps_per_epoch = len(train_images)/128)

model1 = tf.keras.models.load_model('./checkpoints/cp.ckpt')
model1.summary()
result = model1.evaluate(x=test_images, y=test_labels, batch_size=32)
