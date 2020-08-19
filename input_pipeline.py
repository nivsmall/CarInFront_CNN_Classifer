import tensorflow as tf
import cv2


def read_tfrecord(record_file, show=False):
    """
    read tf.record file, to be loaded as a tf Dataset @ training
    :param record_file: tf.record file
    :param show: Bool - for debugging - whether to show images after reading record file
    :return: TensorFlow data.Dataset type
    """
    raw_image_dataset = tf.data.TFRecordDataset(record_file)

    def _parse_image_function(example_proto):
        """
        When reading data as a dataset - each example will hold these features
        """
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string)
        }
        return tf.io.parse_single_example(example_proto, image_feature_description)

    def preprocess_data(example_proto):
        """
        Calling this function (as a Dataset.map) will process each example as follows:
        -decode image from bytes-list to Image tf.Tensor
        -verify data is in correct shape to serve as model input
        -normalize image values from [0,255],int --> [0,1],float32
        :param example_proto:
        :return: (image, class) tuples, meaning dataset will be set as this tuple pairs
        """
        img = tf.io.decode_jpeg(example_proto['image_raw'])
        img = tf.reshape(img, (227, 227, 3))
        img = tf.cast(img, tf.float32)/255.
        label = example_proto['label']
        return img, label

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    print(parsed_image_dataset)

    parsed_image_dataset = parsed_image_dataset.map(preprocess_data)
    print(parsed_image_dataset)

    if show:
        for image, label in parsed_image_dataset:
            cv2.imshow('label: ' + str(label), image.numpy())
            stop = chr(cv2.waitKey())
            cv2.destroyAllWindows()
            if stop == '9':
                break

    return parsed_image_dataset
