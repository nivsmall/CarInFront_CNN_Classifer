import tensorflow as tf
import cv2


def read_tfrecord(record_file, batch_size=1, shuffle=True, num_epochs=1, show=False):
    raw_image_dataset = tf.data.TFRecordDataset(record_file)

    def _parse_image_function(example_proto):
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string)
        }
        return tf.io.parse_single_example(example_proto, image_feature_description)

    def preprocess_data(example_proto):
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