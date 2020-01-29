import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images = np.expand_dims(train_images, axis = -1)
test_images = np.expand_dims(test_images, axis = -1)

inputLayer = tf.layers.Input(shape = (28, 28, 1))
x = inputLayer

for i in range(2):
    x = tf.keras.layers.Conv2D(filters = 32*2**i, kernel_size = 3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.layers.Dense(512)(x, activation = 'relu')
output = tf.layers.Dense(10)(x, activation = 'softmax')

model = tf.keras.model(inputs = inputLayer, outputs = output)
model.compile('adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_crossentropy', 'accuracy'])
model.fit(x=train_images, y=train_labels, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.15, shuffle=True)
result = model.evaluate(x=test_images, y=test_labels, batch_size=32)
print('Test: sparse_categorical_crossentropy, accuracy': result)

