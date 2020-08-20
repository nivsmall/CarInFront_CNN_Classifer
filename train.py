
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

from alexnet_model_seq import create_alexnet
from data_processing import load_labels
import input_pipeline

EPOCHS = 3
INPUT_SHAPE = (227, 227, 3)
NUM_CLASSES = 1
BATCH_SIZE = 16
lr = 1e-2


def train(transfer=False):
    """
    Training the model will be done here!
    :param transfer: Bool- whether to start training from scratch (initialize model) or from pre-trained model
    :return: None
    """

    model = create_alexnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, drop_out_rate=0)
    if transfer:
        model.load_model("training_1/cp-0005.ckpt")

    train_ds = input_pipeline.read_tfrecord('bdd100k/tfrecords/train_test.tfrecords', show=False)
    train_ds = train_ds.shuffle(buffer_size=2048).batch(BATCH_SIZE)

    val_ds = input_pipeline.read_tfrecord('bdd100k/tfrecords/val_test.tfrecords', show=False)
    val_ds = val_ds.shuffle(buffer_size=128).batch(BATCH_SIZE)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="saved_models/my_model2/",
                                                    save_best_only=True,
                                                    monitor='loss',
                                                    save_weights_only=False,
                                                    verbose=1, period=1)]
    '''[tf.keras.callbacks.EarlyStopping(
         monitor="val_loss",  # Stop training when `val_loss` is no longer improving
         min_delta=1e-5,  # "no longer improving" being defined as "no better than 1e-3 less"
         patience=4,  # "no longer improving" being further defined as "for at least # epochs"
         verbose=1,)
     ]'''

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    val_loss, val_acc = model.evaluate(val_ds, verbose=2)

    return


#train(False)
