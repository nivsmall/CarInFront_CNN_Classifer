import tensorflow as tf
from alexnet_model_seq import create_alexnet
import cv2
import os


INPUT_SHAPE = (227, 227, 3)
NUM_CLASSES = 2


def predict(image_dir, saved_model_path, show=False):
    model = tf.keras.models.load_model(saved_model_path)
    ls = os.scandir(image_dir)
    y_lst = []
    for img in ls:
        x = cv2.imread(os.path.join(image_dir, img.name))
        x1 = x
        x = x/255.
        x = tf.image.resize_with_pad(x, target_height=INPUT_SHAPE[0], target_width=INPUT_SHAPE[1])
        if x.numpy().shape != INPUT_SHAPE:
            raise ValueError('The input image shape x doesnt match the model input shape')
        x = tf.expand_dims(x, 0)
        y = model(x, training=False)
        if show:
            cv2.imshow('Model Classified this as: {}'.format(y), x1)
            cv2.waitKey()
            cv2.destroyAllWindows()
        y_lst.append((img.name, y.numpy()[0, 0]))

    return y_lst

#CarInFront = predict('bdd100k/used_images/test/', 'saved_models/my_model3', True)
#print(CarInFront)