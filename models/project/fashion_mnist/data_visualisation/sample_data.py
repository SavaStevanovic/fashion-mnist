import tensorflow as tf
import cv2

def get_validation_sample(validation_data):
    feature_batch = validation_data.take(1)
    for images, labels in feature_batch:
        show(images, labels, 'validation')

def get_training_sample(training_data):
    feature_batch = training_data.take(1)
    for images, labels in feature_batch:
        show(images, labels, 'train')


def show(images, labels, name):
    grid_image = tf.concat([tf.concat([images[i*10 + j] for j in range(10)], 1) for i in range(len(images)//10)], 0)
    grid_image = tf.squeeze(grid_image)
    cv2.imwrite( name+"_sample.png", grid_image.numpy()*255)