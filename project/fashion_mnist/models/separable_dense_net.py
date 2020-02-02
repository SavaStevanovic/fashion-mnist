import tensorflow as tf

def get_model():
    inputLayer = tf.keras.layers.Input(shape = (28, 28, 1))
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='SAME', use_bias=False)(inputLayer)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)

    rens_net_blocks = 3
    for i in range(rens_net_blocks):
        net = dense_block(net, growth_rate = 32)
        if i+1!=rens_net_blocks:
            net = transition(net)
        
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.AveragePooling2D()(net)
    net = tf.keras.layers.Flatten()(net)
    output = tf.keras.layers.Dense(10, activation = 'softmax')(net)

    return tf.keras.Model(inputs = inputLayer, outputs = output, name = 'separable_densenet_model')

def transition(net):
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Conv2D(filters=net.shape[-1].value//2, kernel_size=1, padding='SAME', use_bias=False)(net)
    net = tf.keras.layers.AveragePooling2D()(net)
    return net

def bottleneck_layer(net, filters):
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.SeparableConv2D(filters=4*filters, kernel_size=1, padding='SAME', use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, padding='SAME', use_bias=False)(net)
    return net

def dense_layer(net, filters):
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, padding='SAME', use_bias=False)(net)
    return net

def dense_block(net, growth_rate):
    net0 = net
    net1 = dense_layer(net, growth_rate)
    net2 = dense_layer(tf.keras.layers.concatenate([net1, net], -1), growth_rate)
    net3 = dense_layer(tf.keras.layers.concatenate([net2, net1, net], -1), growth_rate)
    net4 = dense_layer(tf.keras.layers.concatenate([net3, net2, net1, net], -1), growth_rate)
    net5 = tf.keras.layers.concatenate([net4, net3, net2, net1, net0], -1)
    return net5

