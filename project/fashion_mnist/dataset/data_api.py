import tensorflow as tf
import multiprocessing

def get_train_data(data, labels):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(data))
    dataset = dataset.repeat()
    dataset = dataset.map(augment_data, num_parallel_calls = multiprocessing.cpu_count())
    dataset = dataset.batch(128)

    return dataset

@tf.function
def augment_data(image, label):
    if label in [0, 1, 2, 3, 4, 6, 8]:
        image = tf.image.random_flip_left_right(image)

    return image, label

def get_validation_data(data, labels):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(128)

    return dataset
