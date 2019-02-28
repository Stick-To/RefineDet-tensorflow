from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import RefineDet320 as net
import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io, transform
# from utils.voc_classname_encoder import classname_to_ids
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
lr = 0.01
batch_size = 32
buffer_size = 512
epochs = 300
reduce_lr_epoch = [50, 200]
ckpt_path = os.path.join('.', 'vgg_16.ckpt')
config = {
    'mode': 'train',  # train ,test
    'data_format': 'channels_last',
    'num_classes': 20,
    'weight_decay': 5e-4,
    'keep_prob': 0.5,  # not used
    'batch_size': batch_size,
    'nms_score_threshold': 0.5,
    'nms_max_boxes': 20,
    'nms_iou_threshold': 0.5,
    'pretraining_weight': ckpt_path
}

image_preprocess_config = {
    'data_format': 'channels_last',
    'target_size': [320, 320],
    'shorter_side': 480,
    'is_random_crop': False,
    'random_horizontal_flip': 0.5,
    'random_vertical_flip': 0.,
    'pad_truth_to': 60
}

data = ['./test/test_00000-of-00005.tfrecord',
        './test/test_00001-of-00005.tfrecord']

train_gen = voc_utils.get_generator(data,
                                    batch_size, buffer_size, image_preprocess_config)
trainset_provider = {
    'data_shape': [320, 320, 3],
    'num_train': 5000,
    'num_val': 0,
    'train_generator': train_gen,
    'val_generator': None
}
refinedet = net.RefineDet320(config, trainset_provider)
# refinedet.load_weight('./refinedet320/test-64954')
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = refinedet.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    refinedet.save_weight('latest', './refinedet320/test')



# img = io.imread('000026.jpg')
# img = transform.resize(img, [300,300])
# img = np.expand_dims(img, 0)
# result = ssd300.test_one_image(img)
# id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
# scores = result[0]
# bbox = result[1]
# class_id = result[2]
# print(scores, bbox, class_id)
# plt.figure(1)
# plt.imshow(np.squeeze(img))
# axis = plt.gca()
# for i in range(len(scores)):
#     rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
#     axis.add_patch(rect)
#     plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
# plt.show()
