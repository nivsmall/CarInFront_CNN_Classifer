
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import datetime

from alexnet_model_seq import create_alexnet
from data_processing import load_labels
import input_pipeline

EPOCHS = 40
INPUT_SHAPE = (227, 227, 3)
NUM_CLASSES = 1
BATCH_SIZE = 16
lr = 1e-6
drop_out = 0
batchNorm = False


def train(tfrecord_path, transfer=False):
    """
    Training the model will be done here!
    :param tfrecord_path: train dataset tfrecord path
    :param transfer: Bool- whether to start training from scratch (initialize model) or from pre-trained model
    :return: None
    """

    start_time = datetime.datetime.now().strftime("%d_%m_%Y %H_%M")

    if transfer:
        model = tf.keras.models.load_model("saved_models/my_model1")
    else:
        model = create_alexnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, b_n=batchNorm, drop_out_rate=drop_out)

    train_ds = input_pipeline.read_tfrecord(tfrecord_path, show=False)
    train_ds = train_ds.shuffle(buffer_size=16384).batch(BATCH_SIZE)

    val_ds = input_pipeline.read_tfrecord('bdd100k/tfrecords/val.tfrecords', show=False)
    val_ds = val_ds.shuffle(buffer_size=2048).batch(BATCH_SIZE)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="saved_models/my_model_{}/".format(start_time),
                                                    save_best_only=True,
                                                    monitor='val_loss',
                                                    save_weights_only=False,
                                                    verbose=1, period=1),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  min_delta=0,
                                                  patience=15,
                                                  verbose=1,)]

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

    print(history)

    val_loss, val_acc = model.evaluate(val_ds, verbose=2)
    plot_training_session(history, lr, drop_out, batchNorm, val_loss, val_acc, start_time)

    return


def plot_training_session(history, l_r, d_o, b_n, val_loss, val_acc, start_time):

    if b_n: b_n = 'b_n'
    else: b_n = ''

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Training Session: {0}\n Val: loss-acc = {1}-{2}'.format(start_time, val_loss, val_acc))
    ax1.plot(history.history['accuracy'], label='accuracy')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy')
    ax1.set(xlabel='Epoch', ylabel='Accuracy')
    ax1.legend(loc='lower right')

    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set(xlabel='Epoch', ylabel='Loss')
    ax2.legend(loc='lower right')
    fig.savefig('{0} training -- (lr-{1}) {3} (dr_o-{2}%).png'.format(start_time, l_r, int(100*d_o), b_n))

    return

