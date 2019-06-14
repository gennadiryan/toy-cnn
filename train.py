from keras.preprocessing.image import ImageDataGenerator

from models import models
from param import param

p = param()

scratch = models().scratch_net()
scratch.summary()

# train_data_generator = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=.2,
#     height_shift_range=.2,
#     shear_range=.2,
#     zoom_range=.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

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
        scratch.load_weights(p.weight)
        print('Loaded!')
except:
    print('Failed to load old weights')

scratch.fit_generator(
    train_generator,
    steps_per_epoch=2000 // p.batch_size,
    epochs=5,
    validation_data=test_generator,
    validation_steps=800 // p.batch_size
)

scratch.save_weights('try1.h5')