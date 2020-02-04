import tensorflow as tf
import os
import numpy as np
import time

def inferenceTime(model, data, name):
    ts = time.time()
    for image in data:
        _ = model.predict(image, workers=8, use_multiprocessing=True)
    te = time.time()
    print('%r  %2.4f s' % (name, (te - ts)))

_, (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
test_images = np.expand_dims(test_images, axis = -1)/255
test_images = np.expand_dims(test_images, axis = 1)

for folder in os.listdir('./models_and_logs'):
    time.sleep(500)
    model_path = os.path.join('./models_and_logs', folder, 'cp.ckpt')
    test_model = tf.keras.models.load_model(model_path)
    inferenceTime(test_model, test_images, folder)