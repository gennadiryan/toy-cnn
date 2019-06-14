from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from param import param

p = param()


def tfrecord_generator(path):
    global p

    def parser(record):
        features = {'image': tf.FixedLenFeature([], tf.string),
                         'label': tf.FixedLenFeature([], tf.int64)}

        parsed = tf.parse_single_example(record, features)

        parsed['image'] = tf.decode_raw(parsed['image'], tf.float32)
        parsed['image'] = tf.cast(parsed['image'], tf.float32)

        return parsed['image'], parsed['label']

    set = tf.data.TFRecordDataset([path])
    set = set.map(parser)
    set = set.repeat()

    set = set.batch(p.batch_size)
    # set.shuffle(p.buffer_size)

    gen = set.make_one_shot_iterator()

    sess = tf.Session()
    while True:
        yield sess.run(gen.get_next())

gen = tfrecord_generator('/Users/gennadiryan/Downloads/cat_dog_test.tfrecord')

def show_batch_head(batch):
    images, labels = batch
    image, label = images[0], labels[0]

    # image = image.astype(int)
    image = np.reshape(image, (300, 300, 3))

    plt.imshow(image)
    plt.title(str(label))
    plt.show()

for batch in gen: show_batch_head(batch)