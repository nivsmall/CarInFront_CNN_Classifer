import tensorflow as tf
import numpy as np
import json
import os
import cv2
import random


def load_labels(lbl_dir):
    """
    load Json file and convert it to a python dictionary
    :param lbl_dir: directory of Json file including file name
    :return: dictionary containing { KEY - image name, VALUE - image data }
    """
    data_dict = {}
    with open(lbl_dir) as json_file:
        labels_dict = json.load(json_file)
    return labels_dict


def image_example(img_str, label):
    """
    When tf.record file is written (as a batch) this function will determine what data to write to each example
    :param img_str: a bytes-string encoded image
    :param label: class of example
    :return:
    """

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    image_shape = tf.image.decode_jpeg(img_str).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(img_str),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(record_file, labels_dict, img_dir, single_img_folder, In_Shape=(227, 227, 3)):
    """
    :param record_file: tf.record type file to serve as input for training model
    :param labels_dict: dictionary containing { KEY - image name, VALUE - class } pairs
    :param img_dir: Image directory - actual images held in '0' and '1' folders
    :param In_Shape: shape of images to serve as inputs to model
    :return: int - number of examples written in the tf.record file
        Notice: the only pre-processing done here is resizing the images
                Normalizing the images will be done elsewhere; @ input_pipeline.py
    """
    if single_img_folder:   # all images in single folder
        img_dir0 = img_dir1 = img_dir
    else:                   # different labels are stored are stored in different sub folders:
        img_dir0 = os.path.join(img_dir, '0')
        img_dir1 = os.path.join(img_dir, '1')

    record_dir = os.path.split(record_file)[0]
    if not os.path.isdir(record_dir): os.mkdir(record_dir)

    with tf.io.TFRecordWriter(record_file) as writer:
        i = 0
        for filename, label in labels_dict.items():
            label = int(label)
            if label == 0:
                img_dir = img_dir0
            elif label == 1:
                img_dir = img_dir1
            else:
                continue

            image = cv2.imread(os.path.join(img_dir, filename))
            # resize to model input size:
            image = tf.cast(tf.image.resize_with_pad(image, target_height=In_Shape[0], target_width=In_Shape[1]), tf.uint8)
            image_string = tf.io.encode_jpeg(tf.cast(image, tf.uint8))

            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())
            i += 1
    return i


def balance_data_via_augmentation(data_dir, balance_only):
    """
    Assuming there are more One labels than Zeros (but no more than half):
    Augment all Zero-labeled data
    Augment One-labeled data to achieve equal number of images for both classes
    Amount of data after running this function will be 2*0-class images
    :param data_dir: directory that image data is held, Image data should already be split into folders for each class
    :param balance_only: Bool -- True - each image gets augmented once (blur or noise or flip)
                                 False - each image augmented thrice (blur and noise and flip, separately)
                                Either way this data should be balanced at end of run (4 or 8 times zero labeled)
    :return: None
    """
    img_dir0 = os.path.join(data_dir, '0')
    img_dir1 = os.path.join(data_dir, '1')
    list0 = os.listdir(img_dir0)
    list1 = os.listdir(img_dir1)
    if balance_only:
        data_imbalance = len(list0)*2-len(list1)
        to_augment = random.choices(list1, k=data_imbalance)
    else:
        data_imbalance = len(list0)-len(list1)
        data_imbalance = len(list0)-int(data_imbalance/3)
        to_augment = random.choices(list1, k=data_imbalance)
    random.shuffle(to_augment)
    random.shuffle(list0)
    augment_listed_images(img_dir1, to_augment, balance_only)
    augment_listed_images(img_dir0, list0, balance_only)
    return


def augment_listed_images(img_dir, img_lst, balance_only):
    """
    Augments images and writes to same directory, types of augmentation:
    1) flip horizontally 2) add random noise 3) add gaussian blur
    :param img_dir: directory of images
    :param img_lst: list type containing all images to augment - all images should be in img_dir
    :param balance_only: Bool
    :return: None
    """
    for i in range(len(img_lst)):

        image_path = os.path.join(img_dir, img_lst.pop())
        image = cv2.imread(image_path)

        if not balance_only:
                # add noise:
            aug_img = add_random_noise(image)
            cv2.imwrite(image_path.split('.jpg')[0] + '_n_aug.jpg', aug_img)
                # flip horizontally:
            aug_img = np.fliplr(image)
            cv2.imwrite(image_path.split('.jpg')[0] + '_f_aug.jpg', aug_img)
                # add blur:
            aug_img = blur_image(image)
            cv2.imwrite(image_path.split('.jpg')[0] + '_b_aug.jpg', aug_img)

        elif i % 3 == 0:  # add noise
            aug_img = add_random_noise(image)
            cv2.imwrite(image_path.split('.jpg')[0] + '_n_aug.jpg', aug_img)

        elif i % 3 == 1:  # flip horizontally:
            aug_img = np.fliplr(image)
            cv2.imwrite(image_path.split('.jpg')[0] + '_f_aug.jpg', aug_img)

        elif i % 3 == 2:  # add blur
            aug_img = blur_image(image)
            cv2.imwrite(image_path.split('.jpg')[0] + '_b_aug.jpg', aug_img)
    return


def add_random_noise(image, show=False):
    """
    Adds random noise to image for augmentation
    :param image: array - Image Matrix
    :param show: Bool - for debugging - whether to show image after added noise
    :return Augmented Image - array - Image Matrix
    """
    std = np.random.randint(10, 50, 1)
    aug_img = np.zeros(image.shape, np.uint8)
    aug_img = np.clip(cv2.randn(aug_img, mean=255 / 2, stddev=std), a_min=0, a_max=255)
    aug_img = cv2.add(aug_img, image)
    if show:
        cv2.imshow('Augmented Image:', aug_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return aug_img


def blur_image(image):
    """
    Blurs image, Gaussian Blur
    :param image: array - Image Matrix
    :return: Augmented Image - array - Image Matrix
    """
    kernel = random.randint(5, 50)
    if kernel % 2 == 0: kernel += 1
    aug_img = cv2.GaussianBlur(image, ksize=(kernel, kernel), sigmaX=0)
    return aug_img






