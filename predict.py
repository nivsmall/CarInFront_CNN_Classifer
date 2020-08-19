import tensorflow as tf
from alexnet_model_seq import create_alexnet
import cv2
import os


INPUT_SHAPE = (227, 227, 3)
NUM_CLASSES = 2


def predict(image_dir, saved_model_path):
    model = tf.keras.models.load_model(saved_model_path)
    ls = os.scandir(img_dir)
    y_lst = []
    for img in ls:
        x = cv2.imread(os.path.join(image_dir, img.name))
        #cv2.imshow('Image to Classify:', x)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        x = x/255.
        x = tf.image.resize_with_pad(x, target_height=INPUT_SHAPE[0], target_width=INPUT_SHAPE[1])
        if x.numpy().shape != INPUT_SHAPE:
            raise ValueError('The input image x doesnt match the model input')
        x = tf.expand_dims(x, 0)
        y = model(x, training=False)

        y_lst.append(y.numpy()[0, 0])

    return y_lst

img_dir = 'bdd100k/used_images/test/'
CarInFront = predict(img_dir, 'saved_models/my_model')
print(CarInFront)

#CarInFront = predict('bdd100k/used_images/test/cb86b1d9-7735472c.jpg', 'saved_models/my_model')
#print(CarInFront.numpy())