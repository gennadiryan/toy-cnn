from keras.preprocessing.image import ImageDataGenerator

from models import models
from param import param

p = param()

vgg = models().vgg_net_v2()
vgg.summary()

train_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True
)

train_generator = train_data_generator.flow_from_directory(
    p.train_fold,
    target_size=(150, 150),
    batch_size=p.batch_size,
    class_mode='binary'
)

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    p.test_fold,
    target_size=(150, 150),
    batch_size=p.batch_size,
    class_mode='binary'
)

try:
    with open(p.weight, 'r') as f:
        vgg.load_weights('/Users/gennadiryan/PycharmProjects/toy-cnn/reuse-vgg_4.h5') # old weights
        print('Loaded!')
except:
    print('Failed to load old weights')
    exit()

for i, layer in enumerate(vgg.layers[::-1]):
    layer.trainable = True
    if type(layer).__name__ == 'Conv2D':
        print('Stopped looking for conv layers after {} iterations.'.format(i + 1))
        break


vgg.fit_generator(
    train_generator,
    steps_per_epoch=2000 // p.batch_size,
    epochs=1,
    validation_data=test_generator,
    validation_steps=800 // p.batch_size
)

vgg.save_weights('reuse-vgg_5.h5')