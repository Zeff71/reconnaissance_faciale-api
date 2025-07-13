import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
tf.compat.v1.disable_eager_execution()

logger = logging.getLogger(__name__)


def read_data(image_paths, label_list, image_size, batch_size, max_nrof_epochs, num_threads, shuffle, random_flip,
              random_brightness, random_contrast):
    """
    Creates Tensorflow Queue to batch load images. Applies transformations to images as they are loaded.
    :param random_brightness: 
    :param random_flip: 
    :param image_paths: image paths to load
    :param label_list: class labels for image paths
    :param image_size: size to resize images to
    :param batch_size: num of images to load in batch
    :param max_nrof_epochs: total number of epochs to read through image list
    :param num_threads: num threads to use
    :param shuffle: Shuffle images
    :param random_flip: Random Flip image
    :param random_brightness: Apply random brightness transform to image
    :param random_contrast: Apply random contrast transform to image
    :return: images and labels of batch_size
    """

    images = ops.convert_to_tensor(image_paths, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

    # Makes an input queue

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.repeat(max_nrof_epochs)

    parse_function = get_parse_function(image_size, random_flip, random_brightness, random_contrast)
    dataset = dataset.map(parse_function, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

def get_parse_function(image_size, random_flip, random_brightness, random_contrast):
    def parse_function(filename, label):
        file_contents = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(file_contents, channels=3)
        image = tf.image.random_crop(image, size=[image_size, image_size, 3])
        image = tf.image.per_image_standardization(image)

        if random_flip:
            image = tf.image.random_flip_left_right(image)
        if random_brightness:
            image = tf.image.random_brightness(image, max_delta=0.3)
        if random_contrast:
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        return image, label
    return parse_function

def read_image_from_disk(filename_to_label_tuple):
    """
    Consumes input tensor and loads image
    :param filename_to_label_tuple: 
    :type filename_to_label_tuple: list
    :return: tuple of image and label
    """
    label = filename_to_label_tuple[1]
    file_contents = tf.io.read_file(filename_to_label_tuple[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label


def get_image_paths_and_labels(dataset):
    image_paths = []
    labels = []
    label_dict = {}
    label_counter = 0

    for class_obj in dataset:
        class_name = class_obj.name
        if class_name not in label_dict:
            label_dict[class_name] = label_counter
            label_counter += 1

        label = label_dict[class_name]

        for image_path in class_obj.image_paths:
            image_paths.append(image_path)
            labels.append(label)

    return image_paths, np.array(labels)



def get_dataset(input_directory):
    dataset = []

    classes = os.listdir(input_directory)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(input_directory, class_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            image_paths = [os.path.join(facedir, img) for img in images]
            dataset.append(ImageClass(class_name, image_paths))

    return dataset


def filter_dataset(dataset, min_images_per_label=10):
    filtered_dataset = []
    for i in range(len(dataset)):
        if len(dataset[i].image_paths) < min_images_per_label:
            logger.info('Skipping class: {}'.format(dataset[i].name))
            continue
        else:
            filtered_dataset.append(dataset[i])
    return filtered_dataset


def split_dataset(dataset, split_ratio=0.8):
    train_set = []
    test_set = []
    min_nrof_images = 2
    for cls in dataset:
        paths = cls.image_paths
        np.random.shuffle(paths)
        split = int(round(len(paths) * split_ratio))
        if split < min_nrof_images:
            continue  # Not enough images for test set. Skip class...
        train_set.append(ImageClass(cls.name, paths[0:split]))
        test_set.append(ImageClass(cls.name, paths[split:-1]))
    return train_set, test_set


class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)
