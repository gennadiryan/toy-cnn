from glob import glob
import os
import random

import cv2
import imageio
from tensorflow.keras import backend as K
# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np

import param

p = param.param()
rand = random.Random(42)

assert K.image_data_format() == 'channels_last'

def preprocess_input(imgs):
    batch = []
    for img in imgs:
        img = np.array(img, dtype=float)
        img -= np.amin(img)
        img /= np.amax(img)
        batch.append(img)

    batch = np.array(batch)
    return batch

def get_img(path, preview=False):
    img = image.load_img(path, target_size=(224, 224))
    img = image.img_to_array(img)

    if preview:
        plt.imshow(img)
        plt.show()

    return img

# Settings
# imgs_cats = glob(os.path.join(p.test_fold, 'cats', '*.tif'))
# imgs_dogs = glob(os.path.join(p.test_fold, 'dogs', '*.tif'))
# rand.shuffle(cats)
# rand.shuffle(dogs)
# imgs = glob(os.path.join(p.test_fold, '**/*.jpg'))
# rand.shuffle(imgs)

overlay_path = './neuron_grads/overlays'
heatmap_path = './neuron_grads/heatmaps'

dir = '/Users/gennadiryan/Documents/keras'
model_f = os.path.join(dir, 'model.h5')
imgs = glob(os.path.join(dir, 'imgs', '*.tif'))[9:10]
print(imgs)

img_list = list(map(lambda path: get_img(path, preview=False), imgs))
lbls = np.array(list(map(lambda path: int('live' in path), imgs)))

batch = np.array(img_list)
batch = preprocess_input(batch)
print(batch.shape)
print(lbls, len(lbls))

# instantiate model after preliminary setups done
# model = VGG16(weights='imagenet')
model = load_model(model_f)
# print(model.evaluate(batch, lbls, batch_size=20, verbose=1))
# exit()

preds = model.predict(batch)
# print(decode_predictions(preds, top=1)[0])
top_class = np.argmax(preds[0])
# top_class = np.argsort(-preds[0])[2]
print(preds)

layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]

pred_vector = model.output[:, top_class]

heatmaps = []
for layer_name in layer_names:
    cur_layer = model.get_layer(layer_name)
    grads = K.gradients(pred_vector, cur_layer.output)[0]
    pooled = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled, cur_layer.output[0]])
    pooled_val, conv_val = iterate([batch])

    # iterates over filters (channels)
    for i in range(model.get_layer(layer_name).output_shape[-1]):
        conv_val[:, :, i] *= pooled_val[i]

    cur_map = np.mean(conv_val, axis=-1)
    cur_map -= np.amin(cur_map)
    cur_map /= np.amax(cur_map)
    cur_map = cv2.resize(cur_map, (224, 224))
    cur_map = np.uint8(cur_map * 255)

    heatmaps.append(cur_map)
    # print(cur_map.shape)
    # plt.imshow(cur_map)
    # plt.title(layer_name)
    # plt.show()

    cv2.imwrite(os.path.join(heatmap_path, 'grad_{}.jpg'.format(layer_name)), cur_map)

# exit()

img = img_list[0]
# plt.imshow(img)
# plt.show()

for i, (layer_name, hm) in enumerate(zip(layer_names, heatmaps)):
    cur_map = cv2.resize(hm, (224, 224))
    # cur_map = np.uint8(cur_map * 255)
    cur_map = cv2.applyColorMap(cur_map, cv2.COLORMAP_JET)

    overlay = cur_map * .4 + img

    cv2.imwrite(os.path.join(overlay_path, 'grad_{}.jpg'.format(layer_name)), overlay)

    # plt.imshow(overlay)
    # plt.show()

