from glob import glob
import os
import random

import json
import matplotlib.pyplot as plt
import numpy as np

# data = []
# with open('block1_conv1_filters.json') as fh:
#     data = json.load(fh)

json_dir = 'filter_grads'

files = glob(os.path.join(json_dir, '*_filters.json'))
file_opts = list(map(lambda s: s.split('/')[-1].replace('_filters.json', ''), files))

print('Convolution layer options: ')
for i, opt in enumerate(file_opts):
    print('{} - {}'.format(i, opt))

# First selection (for layer)
sel = -1
tried = False
while sel < 0 or sel >= len(files):
    print('' if tried is False else 'Options must be between 0 and {}'.format(len(files) - 1))
    tried = True
    print()
    try:
        sel = int(input('Select an option from the above: '))
    except ValueError:
        continue

print('Picked layer {}'.format(file_opts[sel]))

data = []
with open(files[sel]) as fh:
    data = json.load(fh)

num_filts = len(data)

# while True:
#     sel = input('Pick a filter between 0-{}: '.format(num_filts - 1))
#
#     if sel == 'exit':
#         break
#
#     try:
#         sel = int(sel)
#         if sel < 0 or sel >= num_filts:
#             raise ValueError
#     except ValueError:
#         print('Invalid value entered for filter selection')
#         continue
#
#     print('You picked filter {}'.format(sel))
#
#     img = np.uint8(np.array(data[sel][0]))
#
#     plt.imshow(img)
#     plt.show()
#
# print('Thank you for using!')

enums = list(enumerate(data))
random.shuffle(enums)

for i, (filter, loss) in enums:
    img = np.uint8(np.array(filter))

    plt.imshow(img)
    plt.title('Filter {}; Loss {}'.format(i, loss))
    plt.show()


