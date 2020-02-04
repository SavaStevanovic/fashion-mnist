import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split 
from collections import Counter
import os
import cv2
tf.enable_eager_execution()

def show(images, name):
    grid_image = tf.concat([tf.concat([images[i*10 + j] for j in range(10)], 1) for i in range(len(images)//10)], 0)
    grid_image = tf.squeeze(grid_image)
    cv2.imwrite( name+"_sample.png", grid_image.numpy()*255)

(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
_, validation_images, _, validation_labels = train_test_split(train_images, train_labels, test_size = 0.15, random_state = 42, shuffle = True)
validation_images = np.expand_dims(validation_images, axis = -1)/255

test_model = tf.keras.models.load_model('./models_and_logs/resnet18/cp.ckpt')
predictions_vectors = test_model.predict(x=validation_images, batch_size=128, verbose=0)
predictions = np.argmax(predictions_vectors, 1)

plt.hist(train_labels, 10, histtype='bar', stacked=True, fill=True, rwidth=0.5)
plt.savefig('data_distribution.png')
plt.clf()

plt.hist(validation_labels, 10, histtype='bar', stacked=True, fill=True, rwidth=0.5)
plt.savefig('validation_data_distribution.png')
plt.clf()

confusion_matrix = tf.math.confusion_matrix(validation_labels, predictions, num_classes=10)
sn.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.clf()

diag = np.diagonal(confusion_matrix)
print(diag/np.sum(confusion_matrix))
class_count = Counter(validation_labels)
print(sorted(Counter(validation_labels)))
print(Counter(validation_labels).values())

print([class_count[i]/diag[i] for i in range(10)])

validation_images_array = np.array(validation_images)
wrong_ids = np.logical_and(np.array(validation_labels)==6, np.array(validation_labels)!=np.array(predictions))
show(validation_images_array[wrong_ids], 'wrong')

for folder in os.listdir('./models_and_logs'):
    model_path = os.path.join('./models_and_logs', folder, 'cp.ckpt')
    test_model = tf.keras.models.load_model(model_path)
    predictions_vectors = test_model.predict(x=validation_images, batch_size=128, verbose=0)
    predictions = np.argmax(predictions_vectors, 1)
    confusion_matrix = tf.math.confusion_matrix(validation_labels, predictions, num_classes=10)
    sn.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join('./models_and_logs', folder,'confusion_matrix.png'))
    plt.clf()
    tf.keras.backend.clear_session()
