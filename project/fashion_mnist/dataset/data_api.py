import tensorflow as tf
import multiprocessing

def get_train_data(data, labels):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(data))
    dataset = dataset.repeat()
    dataset = dataset.map(augment_data, num_parallel_calls = multiprocessing.cpu_count())
    dataset = dataset.batch(128)

    return dataset

def augment_data(image, label):
    image = tf.image.random_flip_left_right(image)
    
    lower_bound = tf.random.uniform(shape=(1, 2), minval=0.000, maxval = 1/28)
    upper_bound = tf.random.uniform(shape=(1, 2), minval=27/28, maxval = 1.00)
    image = tf.expand_dims(image, 0)
    image = tf.image.crop_and_resize(image, tf.concat([lower_bound,upper_bound], -1), [0], (28, 28))
    image = tf.squeeze(image, 0)

    return image, label

def get_validation_data(data, labels):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(128)

    return dataset
