import tensorflow as tf

def get_model():
    inputLayer = tf.keras.layers.Input(shape = (28, 28, 1))
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='SAME', use_bias=False)(inputLayer)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)

    rens_net_blocks = 4
    for i in range(rens_net_blocks):
        net = residual_block(net, filters = 64*2**i)
        net = residual_block(net, filters = 64*2**i)
        if i+1==rens_net_blocks:
            net = tf.keras.layers.AveragePooling2D()(net)
        else:
            net = tf.keras.layers.MaxPool2D()(net)


    net = tf.keras.layers.Flatten()(net)
    output = tf.keras.layers.Dense(10, activation = 'softmax')(net)

    return tf.keras.Model(inputs = inputLayer, outputs = output, name = 'separable_resnet_model')
    
def residual_block(net, filters):
    if net.shape[-1].value != filters:
        net_pre = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='SAME')(net)
    else:
        net_pre = net
    net = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, padding='SAME', use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, padding='SAME', use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.add([net, net_pre])
    net = tf.keras.layers.Activation('relu')(net)
    return net

