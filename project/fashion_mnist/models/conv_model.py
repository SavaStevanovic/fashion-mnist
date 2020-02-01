import tensorflow as tf

def get_model():
    inputLayer = tf.keras.layers.Input(shape = (28, 28, 1))
    x = inputLayer

    for i in range(2):
        x = tf.keras.layers.Conv2D(filters = 64*2**i, kernel_size = 3, activation = 'relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(rate = 0.2, seed = 42)(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    output = tf.keras.layers.Dense(10, activation = 'softmax')(x)

    return tf.keras.Model(inputs = inputLayer, outputs = output)