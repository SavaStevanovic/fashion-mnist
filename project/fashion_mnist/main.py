import tensorflow as tf
import numpy as np
from data_visualisation.sample_data import get_training_sample
from data_visualisation.sample_data import get_validation_sample
from fashion_mnist.callbacks.get_callbacks import get_callbacks
from fashion_mnist.dataset.data_api import get_train_data
from fashion_mnist.dataset.data_api import get_validation_data
from sklearn.model_selection import train_test_split 
from models import conv_model
from models import resnet_model
tf.enable_eager_execution()

sample_data = False 

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
test_images = np.expand_dims(test_images, axis = -1)/255
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size = 0.15, random_state = 42, shuffle = True)
training_data = get_train_data(train_images, train_labels)
validation_data = get_validation_data(validation_images, validation_labels)

if sample_data:
    get_training_sample(training_data)
    get_validation_sample(validation_data)

model = resnet_model.get_model()
model.compile('adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_crossentropy', 'accuracy'])
model.fit(x=training_data, validation_data=validation_data, epochs=100, verbose=1, callbacks=get_callbacks(), steps_per_epoch = len(train_images)/128)

test_model = tf.keras.models.load_model('./checkpoints/cp.ckpt')
test_model.evaluate(x=test_images, y=test_labels, batch_size=32, verbose=1)
test_model.summary()
