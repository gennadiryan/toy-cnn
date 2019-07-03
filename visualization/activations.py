from glob import glob
import random
import os

import keras.models
from keras.preprocessing import image
from matplotlib import pyplot as plt
import numpy as np

from models import models as my_models
from param import param

p = param()
rand = random.Random(42)

def normalize(img_arr):
    '''
    Normalize pixel intensities to 0-1 scale
    :param img_arr: numpy image(s) array
    :return: normalized image(s) to 0-1 scale
    '''

    ret = np.array(img_arr, dtype=np.float)
    ret -= np.amin(img_arr)
    ret /= np.amax(img_arr)

    return ret

imgs = glob(os.path.join(p.test_fold, 'cats', '*.jpg'))
rand.shuffle(imgs)

cat = normalize(image.load_img(imgs[0], target_size=(150, 150)))
batch = np.expand_dims(cat, axis=0)

vgg = my_models().vgg_net_v2()
# vgg.summary()
vgg.load_weights('/Users/gennadiryan/PycharmProjects/toy-cnn/reuse-vgg_5.h5')


layer_outs = [layer.get_output_at(0) for layer in vgg.layers[1:]]
activation_mod = keras.models.Model(inputs=vgg.input, outputs=layer_outs)

activations = activation_mod.predict(batch)

for act in activations:
    print(act.shape)

# block = activations[2]
for i, block in enumerate(activations):
    img1 = block[0]
    print('Block {} shape: {}'.format(i, img1.shape))

    filters = np.transpose(img1, (2, 0, 1))
    filters = filters[:64, :, :]
    print(filters.shape)

    filters = filters.reshape((8, 8, filters.shape[1], filters.shape[2]))

    rows = []
    for row in filters:
        rows.append(np.hstack(row))

    grid = np.vstack(np.array(rows))

    plt.imshow(batch[0])
    plt.show()

    plt.imshow(grid)
    plt.show()