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
    image = tf.pad(tensor = image, paddings = tf.constant([[2, 2,], [2, 2]]), mode = "CONSTANT", constant_values = 0, name = 'pad')
    image = tf.image.random_crop(image, (28, 28))
    image, label = preprocess_data(image, label)
    image = tf.image.random_flip_left_right(image)

    return image, label

def preprocess_data(image, labels):
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, -1)/ 255
    return image, labels


def get_validation_data(data, labels):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.map(preprocess_data, num_parallel_calls = multiprocessing.cpu_count())
    dataset = dataset.batch(128)

    return dataset
