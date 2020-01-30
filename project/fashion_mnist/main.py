import tensorflow as tf
import numpy as np
from fashion_mnist.callbacks.get_callbacks import get_callbacks

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = np.expand_dims(train_images, axis = -1)/255
test_images = np.expand_dims(test_images, axis = -1)/255

inputLayer = tf.keras.layers.Input(shape = (28, 28, 1))
x = inputLayer

for i in range(2):
    x = tf.keras.layers.Conv2D(filters = 32*2**i, kernel_size = 3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation = 'relu')(x)
output = tf.keras.layers.Dense(10, activation = 'softmax')(x)

model = tf.keras.Model(inputs = inputLayer, outputs = output)
model.compile('adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_crossentropy', 'accuracy'])
model.fit(x=train_images, y=train_labels, batch_size=128, epochs=50, verbose=1, callbacks=get_callbacks(), validation_split=0.15, shuffle=True)

model1 = tf.keras.models.load_model('./checkpoints/cp.ckpt')
result = model1.evaluate(x=test_images, y=test_labels, batch_size=32)
print('Test:\n Sparse_categorical_crossentropy: {}\n Accuracy: {}'.format(result[1], result[2]))
