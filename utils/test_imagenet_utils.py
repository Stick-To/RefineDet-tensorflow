from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.tfrecord_imagenet_utils as imagenet_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tfrecord = imagenet_utils.dataset2tfrecord('F:\\test\\',
                                           'F:\\tfrecord\\', 'test', 5)
print(tfrecord)
# tfrecord = ['F:\\tfrecord\\test_00001-of-00005.tfrecord',
#             'F:\\tfrecord\\test_00002-of-00005.tfrecord',
#             'F:\\tfrecord\\test_00003-of-00005.tfrecord',
#             'F:\\tfrecord\\test_00004-of-00005.tfrecord',
#             'F:\\tfrecord\\test_00005-of-00005.tfrecord']

# image_preprocess_config = {
#         'data_format': 'channels_last',
#         'target_size': [224, 224],
#         'shorter_side': 256,
#         'is_random_crop': False,
#         'random_horizontal_flip': 0.5,
#         'random_vertical_flip': 0.,
# }
#
# train_gen = imagenet_utils.get_generator(tfrecord, 1, 2, image_preprocess_config)
#
# train_initializer, train_iterator = train_gen
# sess = tf.InteractiveSession()
# sess.run(train_initializer)
# image, label ,shape= train_iterator.get_next()
# img, shape  = sess.run([image, shape])
# print(img.shape,shape)
# img = np.squeeze(img)
# img = np.asarray(img, np.uint8)
# plt.figure()
# plt.imshow(img)
# plt.show()
