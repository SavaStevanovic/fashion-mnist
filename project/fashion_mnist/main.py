import tensorflow as tf
import numpy as np
from fashion_mnist.callbacks.get_callbacks import get_callbacks
from fashion_mnist.dataset.data_api import get_train_data
from fashion_mnist.dataset.data_api import get_validation_data
from sklearn.model_selection import train_test_split 

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = np.expand_dims(train_images, axis = -1)/255
test_images = np.expand_dims(test_images, axis = -1)/255

train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size = 0.15, random_state = 42, shuffle = True)

inputLayer = tf.keras.layers.Input(shape = (28, 28, 1))
x = inputLayer

for i in range(2):
    x = tf.keras.layers.Conv2D(filters = 32*2**i, kernel_size = 3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(rate = 0.25, seed = 42)(x)
x = tf.keras.layers.Dense(256, activation = 'relu')(x)
output = tf.keras.layers.Dense(10, activation = 'softmax')(x)

model = tf.keras.Model(inputs = inputLayer, outputs = output)
model.compile('adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_crossentropy', 'accuracy'])
model.fit(x=get_train_data(train_images, train_labels), validation_data=get_validation_data(validation_images, validation_labels), epochs=100, verbose=1, callbacks=get_callbacks(), steps_per_epoch = len(train_images)/128)

model1 = tf.keras.models.load_model('./checkpoints/cp.ckpt')
model1.summary()
result = model1.evaluate(x=test_images, y=test_labels, batch_size=32)
