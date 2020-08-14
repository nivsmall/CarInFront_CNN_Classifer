import tensorflow as tf
from alexnet_model_seq import create_alexnet

INPUT_SHAPE = (227, 227, 3)
NUM_CLASSES = 2


def predict(x, weights_ckpt_path):
    if x.numpy().shape != INPUT_SHAPE:
        raise ValueError('The input image x doesnt match the model input')
    model = create_alexnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.load_weights(weights_ckpt_path)
    y = model(x, training=False)
    return y
