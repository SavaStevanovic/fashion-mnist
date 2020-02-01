import tensorflow as tf

def get_callbacks():
    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/cp.ckpt', monitor='val_acc', verbose=0, save_best_only=True, save_freq='epoch'))
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, update_freq='epoch', profile_batch=0))

    return callbacks

