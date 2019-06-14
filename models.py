from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend
from keras.applications import vgg16

"""
Returns Sequential nets for image classification
By default all accept (150, 150, 3) input

Todo:
    parameterize input shapes
"""

class models:
    def scratch_net(self):
        def add_relu(model): model.add(Activation('relu'))
        def add_max(model): model.add(MaxPooling2D(pool_size=(2,2)))

        shape = (3, 150, 150) if backend.image_data_format() == 'channels_first' else (150, 150, 3)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=shape))
        add_relu(model)
        add_max(model)

        model.add(Conv2D(32, (3, 3)))
        add_relu(model)
        add_max(model)

        model.add(Conv2D(64, (3, 3)))
        add_relu(model)
        add_max(model)

        model.add(Flatten())
        model.add(Dense(64))
        add_relu(model)
        model.add(Dropout(.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def vgg_net(self):
        model = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input((150, 150, 3)))
        for layer in model.layers:
            layer.trainable = False

        # trainable top layer
        model = Sequential(layers=model.layers) # make sequential; functional by default
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1, activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def vgg_net_v2(self):
        model = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input((150, 150, 3)))
        for layer in model.layers:
            layer.trainable = False

        model = Sequential(layers=model.layers)
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model