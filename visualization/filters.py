from time import time

import json
from keras import backend as K
from keras.applications import vgg16
import numpy as np
from tqdm import tqdm

from models import models as my_mods

# program assumes 'channels_last'

# vgg = my_mods().vgg_net_v2()
# vgg.load_weights('/Users/gennadiryan/PycharmProjects/toy-cnn/reuse-vgg_5.h5')
vgg = vgg16.VGG16(weights='imagenet', include_top=False)

side_len = 150
input_img = vgg.input

layer_dict = dict([(layer.name, layer) for layer in vgg.layers[1:]])


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def gradient_ascent(it, iterations):
    step = 1.

    input = np.random.random((1, side_len, side_len, 3))
    input = (input - .5) * 20 + 150

    for _ in range(iterations):
        loss, grads = it([input])
        input += grads * step

        # print('Current loss: {}'.format(loss))

        if loss <= 0.:
            break

    if loss > 0.:
        img = deprocess_image(input[0])
        # kept_filters.append((img, loss))
        kept_filters.append((img.tolist(), float(loss)))


def filter_loss(filter_idx, layer):
    layer_out = layer_dict[layer].output
    loss = K.mean(layer_out[:, :, :, filter_idx])

    grads = K.gradients(loss, input_img)[0]
    grads = normalize(grads)

    iterate = K.function([input_img], [loss, grads])
    return iterate


kept_filters = []
filters_dict = dict()

layer_dict = {name:layer for name, layer in layer_dict.items() if '4' in name or '5' in name}\

for layer_name in layer_dict.keys():
    if layer_name.split('_')[1][:4] == 'pool':
        continue

    layer = vgg.get_layer(layer_name)
    print('Processing filters in layer {}'.format(layer_name))
    for idx in tqdm(range(min(layer.output.shape[-1], layer.output.shape[-1]))): # was 100 max
        # print('Processing filter {}'.format(idx))

        start = time()
        gradient_ascent(filter_loss(idx, layer_name), 20)

        # print('Processed filter {} in {}s'.format(idx, time() - start))

    fname = '{}_filters.json'.format(layer_name)
    with open(fname, 'w') as fh:
        json.dump(kept_filters, fh)
        print('Filters written to {}'.format(fname))

    filters_dict[layer.name] = kept_filters
    kept_filters = []

for layer_name, kept_filters in filters_dict.items():
    print(layer_name, len(kept_filters))

# with open('filter_dump.json', 'w') as fh:
#     json.dump(filters_dict, fh)