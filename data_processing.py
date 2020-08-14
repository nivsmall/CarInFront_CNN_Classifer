import tensorflow as tf
import numpy as np
import json
import os
import cv2


def load_labels(lbl_dir):
    data_dict = {}
    with open(lbl_dir) as json_file:
        labels_dict = json.load(json_file)
    return labels_dict


def reshape_and_normalize_images(im_dir):
    with os.scandir(im_dir) as scan:
        for img in scan:
            if not '.jpg' in img.name:
                continue
            mat = cv2.imread(os.path.join(im_dir, img.name))
            mat = mat/255


def image_example(img_str, label):

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

    '''
    for line in str(image_example(image_string, label)).split('\n')[:15]:
      print(line)
    print('...')
    '''

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(record_file, labels_dict, img_dir):
    ''''
    Notice: the only pre-processing done here is resizing the images
    '''
    # different labels are stored are stored in different folders:
    img_dir0 = os.path.join(img_dir, '0/')
    img_dir1 = os.path.join(img_dir, '1/')
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
            # resize for model input size:
            image = tf.cast(tf.image.resize_with_pad(image, target_height=227, target_width=227), tf.uint8)
            image_string = tf.io.encode_jpeg(tf.cast(image, tf.uint8))

            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())
            i += 1
    return i







