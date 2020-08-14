from tensorflow.keras import layers, models





def create_alexnet(input_shape, num_classes, drop_out_rate):
    ''''
    initializers: he_normal (kaiming) -- mean: 0 -- std: sqrt(2/fan-in)
    '''
    model = models.Sequential()

    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=4,
                            padding='valid', activation='relu',
                            input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='valid'))

    model.add(layers.Conv2D(256, kernel_size=(5, 5), strides=1,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=1,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal'))

    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=1,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal'))

    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=1,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(layers.Flatten())
    #model.add(layers.Dense(4096, activation='relu', kernel_initializer='he_normal'))
    #model.add(layers.Dense(4096, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dropout(drop_out_rate))
    model.add(layers.Dense(1000, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.summary()
    return model