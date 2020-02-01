import tensorflow as tf

def get_model():
    inputLayer = tf.keras.layers.Input(shape = (28, 28, 1))
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu', padding='SAME', use_bias=False)(inputLayer)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)

    for i in range(3):
        net = residual_block(net, filters = 64*2**i)
        net = residual_block(net, filters = 64*2**i)
        net = tf.keras.layers.MaxPool2D()(net)

    net = tf.keras.layers.Flatten()(net)
    output = tf.keras.layers.Dense(10, activation = 'softmax')(net)

    return tf.keras.Model(inputs = inputLayer, outputs = output, name = 'resnet_model')
    
def residual_block(net, filters, ):
    net_pre = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='SAME', use_bias=False)(net)
    net = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.add([net, net_pre])
    net_pre = net
    net = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.add([net, net_pre])
    net = tf.keras.layers.Activation('relu')(net)
    return net

