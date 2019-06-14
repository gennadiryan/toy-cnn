from glob import glob
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
# from keras.models import load_model

from models import models
from param import param

p = param()

fold = '/Users/gennadiryan/datasets/dogs-vs-cats/test1'
weights = '/users/gennadiryan/PycharmProjects/toy-cnn/reuse-vgg_4.h5'

vgg = models().vgg_net_v2()
vgg.load_weights(weights)

imgs = glob(os.path.join(fold, '*'))
np.random.shuffle(imgs)
imgs = imgs[:10]

animals = ['cat', 'dog']

for img in imgs:
    img = imageio.imread(img)
    img = cv2.resize(img, dsize=(150, 150))

    lbl = vgg.predict_classes(np.array([img]))[0][0]

    plt.imshow(img)
    plt.title(animals[lbl])
    plt.show()