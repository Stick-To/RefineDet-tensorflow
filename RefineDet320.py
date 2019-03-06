from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as wrap
import sys
import os
import numpy as np


class RefineDet320:
    def __init__(self, config, data_provider):
        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider
        self.input_size = 320
        if config['data_format'] == 'channels_last':
            self.data_shape = [320, 320, 3]
        else:
            self.data_shape = [3, 320, 320]
        self.num_classes = config['num_classes'] + 1
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']
        self.data_format = config['data_format']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = len(self.anchor_ratios)
        self.reader = wrap.NewCheckpointReader(config['pretraining_weight'])

        if self.mode == 'train':
            self.num_train = data_provider['num_train']
            self.num_val = data_provider['num_val']
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
            if data_provider['val_generator'] is not None:
                self.val_generator = data_provider['val_generator']
                self.val_initializer, self.val_iterator = self.val_generator

        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.is_training = True

        self._define_inputs()
        self._build_graph()
        self._create_saver()
        if self.mode == 'train':
            self._create_summary()
        self._init_session()

    def _define_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        mean = tf.convert_to_tensor([123.68, 116.779, 103.979], dtype=tf.float32)
        if self.data_format == 'channels_last':
            mean = tf.reshape(mean, [1, 1, 1, 3])
        else:
            mean = tf.reshape(mean, [1, 3, 1, 1])
        if self.mode == 'train':
            self.images, self.ground_truth = self.train_iterator.get_next()
            self.images.set_shape(shape)
            self.images = self.images - mean
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.images = self.images - mean
            self.ground_truth = tf.placeholder(tf.float32, [self.batch_size, None, 5], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_graph(self):
        with tf.variable_scope('feature_extractor'):
            feat1, feat2, feat3, feat4, downsampling_rate1, downsampling_rate2, downsampling_rate3, downsampling_rate4 = self._feature_extractor(self.images)
            anchor_scale1 = downsampling_rate1 * 4
            anchor_scale2 = downsampling_rate2 * 4
            anchor_scale3 = downsampling_rate3 * 4
            anchor_scale4 = downsampling_rate4 * 4
            feat1 = tf.nn.l2_normalize(feat1, axis=3 if self.data_format == 'channels_last' else 1)
            feat2 = tf.nn.l2_normalize(feat2, axis=3 if self.data_format == 'channels_last' else 1)
            feat1_l2_norm = tf.get_variable('feat1_l2_norm', initializer=tf.constant(10.))
            feat1 = feat1_l2_norm * feat1
            feat2_l2_norm = tf.get_variable('feat2_l2_norm', initializer=tf.constant(8.))
            feat2 = feat2_l2_norm * feat2
        with tf.variable_scope('ARM'):
            arm1 = self._arm(feat1, 256, 'arm1')
            arm2 = self._arm(feat2, 256, 'arm2')
            arm3 = self._arm(feat3, 256, 'arm3')
            arm4 = self._arm(feat4, 256, 'arm4')
        with tf.variable_scope('TCB'):
            tcb4 = self._tcb(feat4, 'tcb4')
            tcb3 = self._tcb(feat3, 'tcb3', tcb4)
            tcb2 = self._tcb(feat2, 'tcb2', tcb3)
            tcb1 = self._tcb(feat1, 'tcb1', tcb2)
        with tf.variable_scope('ODM'):
            odm1 = self._odm(tcb1, 256, 'odm1')
            odm2 = self._odm(tcb2, 256, 'odm2')
            odm3 = self._odm(tcb3, 256, 'odm3')
            odm4 = self._odm(tcb4, 256, 'odm4')
        with tf.variable_scope('regressor'):
            if self.data_format == 'channels_first':
                arm1 = tf.transpose(arm1, [0, 2, 3, 1])
                arm2 = tf.transpose(arm2, [0, 2, 3, 1])
                arm3 = tf.transpose(arm3, [0, 2, 3, 1])
                arm4 = tf.transpose(arm4, [0, 2, 3, 1])
                odm1 = tf.transpose(odm1, [0, 2, 3, 1])
                odm2 = tf.transpose(odm2, [0, 2, 3, 1])
                odm3 = tf.transpose(odm3, [0, 2, 3, 1])
                odm4 = tf.transpose(odm4, [0, 2, 3, 1])
            p1shape = tf.shape(odm1)
            p2shape = tf.shape(odm2)
            p3shape = tf.shape(odm3)
            p4shape = tf.shape(odm4)
        with tf.variable_scope('inference'):
            arm1bbox_yx, arm1bbox_hw, arm1conf = self._get_armpred(arm1)
            arm2bbox_yx, arm2bbox_hw, arm2conf = self._get_armpred(arm2)
            arm3bbox_yx, arm3bbox_hw, arm3conf = self._get_armpred(arm3)
            arm4bbox_yx, arm4bbox_hw, arm4conf = self._get_armpred(arm4)
            odm1bbox_yx, odm1bbox_hw, odm1conf = self._get_odmpred(odm1)
            odm2bbox_yx, odm2bbox_hw, odm2conf = self._get_odmpred(odm2)
            odm3bbox_yx, odm3bbox_hw, odm3conf = self._get_odmpred(odm3)
            odm4bbox_yx, odm4bbox_hw, odm4conf = self._get_odmpred(odm4)

            armbbox_yx = tf.concat([arm1bbox_yx, arm2bbox_yx, arm3bbox_yx, arm4bbox_yx], axis=1)
            armbbox_hw = tf.concat([arm1bbox_hw, arm2bbox_hw, arm3bbox_hw, arm4bbox_hw], axis=1)
            armconf = tf.concat([arm1conf, arm2conf, arm3conf, arm4conf], axis=1)
            odmbbox_yx = tf.concat([odm1bbox_yx, odm2bbox_yx, odm3bbox_yx, odm4bbox_yx], axis=1)
            odmbbox_hw = tf.concat([odm1bbox_hw, odm2bbox_hw, odm3bbox_hw, odm4bbox_hw], axis=1)
            odmconf = tf.concat([odm1conf, odm2conf, odm3conf, odm4conf], axis=1)

            a1bbox_y1x1, a1bbox_y2x2, a1bbox_yx, a1bbox_hw = self._get_abbox(anchor_scale1, p1shape, downsampling_rate1)
            a2bbox_y1x1, a2bbox_y2x2, a2bbox_yx, a2bbox_hw = self._get_abbox(anchor_scale2, p2shape, downsampling_rate2)
            a3bbox_y1x1, a3bbox_y2x2, a3bbox_yx, a3bbox_hw = self._get_abbox(anchor_scale3, p3shape, downsampling_rate3)
            a4bbox_y1x1, a4bbox_y2x2, a4bbox_yx, a4bbox_hw = self._get_abbox(anchor_scale4, p4shape, downsampling_rate4)

            abbox_y1x1 = tf.concat([a1bbox_y1x1, a2bbox_y1x1, a3bbox_y1x1, a4bbox_y1x1], axis=0)
            abbox_y2x2 = tf.concat([a1bbox_y2x2, a2bbox_y2x2, a3bbox_y2x2, a4bbox_y2x2], axis=0)
            abbox_yx = tf.concat([a1bbox_yx, a2bbox_yx, a3bbox_yx, a4bbox_yx], axis=0)
            abbox_hw = tf.concat([a1bbox_hw, a2bbox_hw, a3bbox_hw, a4bbox_hw], axis=0)
            if self.mode == 'train':
                i = 0.
                loss = 0.
                cond = lambda loss, i: tf.less(i, tf.cast(self.batch_size, tf.float32))
                body = lambda loss, i: (
                    tf.add(loss, self._compute_one_image_loss(
                        tf.squeeze(tf.gather(armbbox_yx, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(armbbox_hw, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(armconf, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(odmbbox_yx, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(odmbbox_hw, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(odmconf, tf.cast(i, tf.int32))),
                        abbox_y1x1,
                        abbox_y2x2,
                        abbox_yx,
                        abbox_hw,
                        tf.squeeze(tf.gather(self.ground_truth, tf.cast(i, tf.int32))),
                    )),
                    tf.add(i, 1.)
                )
                init_state = (loss, i)
                state = tf.while_loop(cond, body, init_state)
                total_loss, _ = state
                total_loss = total_loss / self.batch_size
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=.9)
                self.loss = total_loss + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
                )
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            else:
                armconf = armconf[0, ...]
                arm_mask = armconf[:, 1] < 0.99
                armbbox_yxt = tf.boolean_mask(armbbox_yx[0, ...], arm_mask)
                armbbox_hwt = tf.boolean_mask(armbbox_hw[0, ...], arm_mask)
                odmbbox_yxt = tf.boolean_mask(odmbbox_yx[0, ...], arm_mask)
                odmbbox_hwt = tf.boolean_mask(odmbbox_hw[0, ...], arm_mask)
                odmconf = tf.boolean_mask(odmconf[0, ...], arm_mask)
                abbox_yxt = tf.boolean_mask(abbox_yx, arm_mask)
                abbox_hwt = tf.boolean_mask(abbox_hw, arm_mask)

                odm_class = tf.argmax(odmconf, axis=-1)
                odm_mask = tf.less(odm_class, self.num_classes - 1)
                armbbox_yxt = tf.boolean_mask(armbbox_yxt, odm_mask)
                armbbox_hwt = tf.boolean_mask(armbbox_hwt, odm_mask)
                odmbbox_yxt = tf.boolean_mask(odmbbox_yxt, odm_mask)
                odmbbox_hwt = tf.boolean_mask(odmbbox_hwt, odm_mask)
                confidence = tf.boolean_mask(odmconf, odm_mask)[:, :self.num_classes - 1]

                abbox_yxt = tf.boolean_mask(abbox_yxt, odm_mask)
                abbox_hwt = tf.boolean_mask(abbox_hwt, odm_mask)
                darm_pbbox_yxt = armbbox_yxt * abbox_hwt + abbox_yxt
                darm_pbbox_hwt = abbox_hwt * tf.exp(armbbox_hwt)
                dodm_pbbox_yxt = odmbbox_yxt * darm_pbbox_hwt + darm_pbbox_yxt
                dodm_pbbox_hwt = darm_pbbox_hwt * tf.exp(odmbbox_hwt)

                dpbbox_y1x1 = dodm_pbbox_yxt - dodm_pbbox_hwt / 2.
                dpbbox_y2x2 = dodm_pbbox_yxt + dodm_pbbox_hwt / 2.
                dpbbox_y1x1y2x2 = tf.concat([dpbbox_y1x1, dpbbox_y2x2], axis=-1)
                filter_mask = tf.greater_equal(confidence, self.nms_score_threshold)
                scores = []
                class_id = []
                bbox = []
                for i in range(self.num_classes - 1):
                    scoresi = tf.boolean_mask(confidence[:, i], filter_mask[:, i])
                    bboxi = tf.boolean_mask(dpbbox_y1x1y2x2, filter_mask[:, i])
                    selected_indices = tf.image.non_max_suppression(

                        bboxi, scoresi, self.nms_max_boxes, self.nms_iou_threshold,
                    )
                    scores.append(tf.gather(scoresi, selected_indices))
                    bbox.append(tf.gather(bboxi, selected_indices))
                    class_id.append(tf.ones_like(tf.gather(scoresi, selected_indices), tf.int32) * i)
                bbox = tf.concat(bbox, axis=0)
                scores = tf.concat(scores, axis=0)
                class_id = tf.concat(class_id, axis=0)
                self.detection_pred = [scores, bbox, class_id]

    def _feature_extractor(self, images):
        conv1_1 = self._load_conv_layer(images,
                                        tf.get_variable(name='kernel_conv1_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv1/conv1_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv1_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv1/conv1_1/biases"),
                                                        trainable=True),
                                        name="conv1_1")
        conv1_2 = self._load_conv_layer(conv1_1,
                                        tf.get_variable(name='kernel_conv1_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv1/conv1_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv1_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv1/conv1_2/biases"),
                                                        trainable=True),
                                        name="conv1_2")
        pool1 = self._max_pooling(conv1_2, 2, 2, name="pool1")

        conv2_1 = self._load_conv_layer(pool1,
                                        tf.get_variable(name='kenrel_conv2_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv2/conv2_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv2_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv2/conv2_1/biases"),
                                                        trainable=True),
                                        name="conv2_1")
        conv2_2 = self._load_conv_layer(conv2_1,
                                        tf.get_variable(name='kernel_conv2_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv2/conv2_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv2_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv2/conv2_2/biases"),
                                                        trainable=True),
                                        name="conv2_2")
        pool2 = self._max_pooling(conv2_2, 2, 2, name="pool2")
        conv3_1 = self._load_conv_layer(pool2,
                                        tf.get_variable(name='kernel_conv3_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv_3_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_1/biases"),
                                                        trainable=True),
                                        name="conv3_1")
        conv3_2 = self._load_conv_layer(conv3_1,
                                        tf.get_variable(name='kernel_conv3_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv3_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_2/biases"),
                                                        trainable=True),
                                        name="conv3_2")
        conv3_3 = self._load_conv_layer(conv3_2,
                                        tf.get_variable(name='kernel_conv3_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_3/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv3_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_3/biases"),
                                                        trainable=True),
                                        name="conv3_3")
        pool3 = self._max_pooling(conv3_3, 2, 2, name="pool3")

        conv4_1 = self._load_conv_layer(pool3,
                                        tf.get_variable(name='kernel_conv4_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv4_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_1/biases"),
                                                        trainable=True),
                                        name="conv4_1")
        conv4_2 = self._load_conv_layer(conv4_1,
                                        tf.get_variable(name='kernel_conv4_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv4_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_2/biases"),
                                                        trainable=True),
                                        name="conv4_2")
        conv4_3 = self._load_conv_layer(conv4_2,
                                        tf.get_variable(name='kernel_conv4_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_3/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv4_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_3/biases"),
                                                        trainable=True),
                                        name="conv4_3")
        pool4 = self._max_pooling(conv4_3, 2, 2, name="pool4")
        conv5_1 = self._load_conv_layer(pool4,
                                        tf.get_variable(name='kernel_conv5_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv5_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_1/biases"),
                                                        trainable=True),
                                        name="conv5_1")
        conv5_2 = self._load_conv_layer(conv5_1,
                                        tf.get_variable(name='kernel_conv5_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv5_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_2/biases"),
                                                        trainable=True),
                                        name="conv5_2")
        conv5_3 = self._load_conv_layer(conv5_2,
                                        tf.get_variable(name='kernel_conv5_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_3/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv5_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_3/biases"),
                                                        trainable=True),
                                        name="conv5_3")
        pool5 = self._max_pooling(conv5_3, 2, 2, 'pool5')
        conv_fc6 = self._conv_layer(pool5, 1024, 3, 1, 'conv_fc6', dilation_rate=2, activation=tf.nn.relu)
        conv_fc7 = self._conv_layer(conv_fc6, 1024, 1, 1, 'conv_fc7', dilation_rate=2, activation=tf.nn.relu)
        conv6_1 = self._conv_layer(conv_fc7, 1024, 1, 2, 'conv6_1', activation=tf.nn.relu)
        conv6_2 = self._conv_layer(conv6_1, 1024, 3, 1, 'conv6_2', activation=tf.nn.relu)

        downsampling_rate1 = 8.0
        downsampling_rate2 = 16.0
        downsampling_rate3 = 32.0
        downsampling_rate4 = 64.0
        return conv4_3, conv5_3, conv_fc7, conv6_2, downsampling_rate1, downsampling_rate2, downsampling_rate3, downsampling_rate4

    def _arm(self, bottom, filters, scope):
        with tf.variable_scope(scope):
            conv1 = self._conv_layer(bottom, filters, 3, 1)
            conv2 = self._conv_layer(conv1, filters, 3, 1)
            conv3 = self._conv_layer(conv2, filters, 3, 1)
            conv4 = self._conv_layer(conv3, filters, 3, 1)
            pred = self._conv_layer(conv4, (2+4)*self.num_anchors, 3, 1)
            return pred

    def _odm(self, bottom, filters, scope):
        with tf.variable_scope(scope):
            conv1 = self._conv_layer(bottom, filters, 3, 1)
            conv2 = self._conv_layer(conv1, filters, 3, 1)
            conv3 = self._conv_layer(conv2, filters, 3, 1)
            conv4 = self._conv_layer(conv3, filters, 3, 1)
            pred = self._conv_layer(conv4, (self.num_classes+4)*self.num_anchors, 3, 1)
            return pred

    def _get_armpred(self, pred):
        pred = tf.reshape(pred, [self.batch_size, -1, 2+4])
        pconf = tf.nn.softmax(pred[..., :2])
        pbbox_yx = pred[..., 2:2+2]
        pbbox_hw = pred[..., 2+2:]
        return pbbox_yx, pbbox_hw, pconf

    def _get_odmpred(self, pred):
        pred = tf.reshape(pred, [self.batch_size, -1, self.num_classes+4])
        pconf = tf.nn.softmax(pred[..., :self.num_classes])
        pbbox_yx = pred[..., self.num_classes:self.num_classes+2]
        pbbox_hw = pred[..., self.num_classes+2:]
        return pbbox_yx, pbbox_hw, pconf

    def _tcb(self, feat, scope, high_level_feat=None):
        with tf.variable_scope(scope):
            conv1 = self._conv_layer(feat, 256, 3, 1, 'conv1', activation=tf.nn.relu)
            conv2 = self._conv_layer(conv1, 256, 3, 1, 'conv2')
            if high_level_feat is not None:
                dconv = self._dconv_layer(high_level_feat, 256, 4, 2, 'dconv')
                conv_sum = tf.nn.relu(conv2 + dconv)
            else:
                conv_sum = tf.nn.relu(conv2)
            conv3 = self._conv_layer(conv_sum, 256, 3, 1, 'conv3', activation=tf.nn.relu)
            return conv3

    def _get_abbox(self, size, pshape, downsampling_rate):
        topleft_y = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
        topleft_x = tf.range(0., tf.cast(pshape[2], tf.float32), dtype=tf.float32)
        topleft_y = tf.reshape(topleft_y, [-1, 1, 1, 1]) + 0.5
        topleft_x = tf.reshape(topleft_x, [1, -1, 1, 1]) + 0.5
        topleft_y = tf.tile(topleft_y, [1, pshape[2], 1, 1]) * downsampling_rate
        topleft_x = tf.tile(topleft_x, [pshape[1], 1, 1, 1]) * downsampling_rate
        topleft_yx = tf.concat([topleft_y, topleft_x], -1)
        topleft_yx = tf.tile(topleft_yx, [1, 1, self.num_anchors, 1])

        anchors = []
        for ratio in self.anchor_ratios:
            anchors.append([size*(ratio**0.5), size/(ratio**0.5)])
        anchors = tf.convert_to_tensor(anchors, tf.float32)
        anchors = tf.reshape(anchors, [1, 1, -1, 2])

        abbox_y1x1 = tf.reshape(topleft_yx - anchors / 2., [-1, 2])
        abbox_y2x2 = tf.reshape(topleft_yx + anchors / 2., [-1, 2])
        abbox_yx = abbox_y1x1 / 2. + abbox_y2x2 / 2.
        abbox_hw = abbox_y2x2 - abbox_y1x1
        return abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw

    def _compute_one_image_loss(self, armbbox_yx, armbbox_hw, armconf,
                                odmbbox_yx, odmbbox_hw, odmconf,
                                abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw, ground_truth):
        slice_index = tf.argmin(ground_truth, axis=0)[0]
        ground_truth = tf.gather(ground_truth, tf.range(0, slice_index, dtype=tf.int64))
        gbbox_yx = ground_truth[..., 0:2]
        gbbox_hw = ground_truth[..., 2:4]
        gbbox_y1x1 = gbbox_yx - gbbox_hw / 2.
        gbbox_y2x2 = gbbox_yx + gbbox_hw / 2.
        class_id = tf.cast(ground_truth[..., 4:5], dtype=tf.int32)
        label = class_id

        abbox_hwti = tf.reshape(abbox_hw, [1, -1, 2])
        abbox_y1x1ti = tf.reshape(abbox_y1x1, [1, -1, 2])
        abbox_y2x2ti = tf.reshape(abbox_y2x2, [1, -1, 2])
        gbbox_hwti = tf.reshape(gbbox_hw, [-1, 1, 2])
        gbbox_y1x1ti = tf.reshape(gbbox_y1x1, [-1, 1, 2])
        gbbox_y2x2ti = tf.reshape(gbbox_y2x2, [-1, 1, 2])
        ashape = tf.shape(abbox_hwti)
        gshape = tf.shape(gbbox_hwti)
        abbox_hwti = tf.tile(abbox_hwti, [gshape[0], 1, 1])
        abbox_y1x1ti = tf.tile(abbox_y1x1ti, [gshape[0], 1, 1])
        abbox_y2x2ti = tf.tile(abbox_y2x2ti, [gshape[0], 1, 1])
        gbbox_hwti = tf.tile(gbbox_hwti, [1, ashape[1], 1])
        gbbox_y1x1ti = tf.tile(gbbox_y1x1ti, [1, ashape[1], 1])
        gbbox_y2x2ti = tf.tile(gbbox_y2x2ti, [1, ashape[1], 1])

        gaiou_y1x1ti = tf.maximum(abbox_y1x1ti, gbbox_y1x1ti)
        gaiou_y2x2ti = tf.minimum(abbox_y2x2ti, gbbox_y2x2ti)
        gaiou_area = tf.reduce_prod(tf.maximum(gaiou_y2x2ti - gaiou_y1x1ti, 0), axis=-1)
        aarea = tf.reduce_prod(abbox_hwti, axis=-1)
        garea = tf.reduce_prod(gbbox_hwti, axis=-1)
        gaiou_rate = gaiou_area / (aarea + garea - gaiou_area)

        best_raindex = tf.argmax(gaiou_rate, axis=1)
        best_abbox_yx = tf.gather(abbox_yx, best_raindex)
        best_abbox_hw = tf.gather(abbox_hw, best_raindex)
        best_armbbox_yx = tf.gather(armbbox_yx, best_raindex)
        best_armbbox_hw = tf.gather(armbbox_hw, best_raindex)
        best_armconf = tf.gather(armconf, best_raindex)
        best_odmbbox_yx = tf.gather(odmbbox_yx, best_raindex)
        best_odmbbox_hw = tf.gather(odmbbox_hw, best_raindex)
        best_odmconf = tf.gather(odmconf, best_raindex)

        best_mask, _ = tf.unique(best_raindex)
        best_mask = tf.contrib.framework.sort(best_mask)
        best_mask = tf.reshape(best_mask, [-1, 1])
        best_mask = tf.sparse.SparseTensor(tf.concat([best_mask, tf.zeros_like(best_mask)], axis=-1),
                                           tf.squeeze(tf.ones_like(best_mask)), dense_shape=[ashape[1], 1])
        best_mask = tf.reshape(tf.cast(tf.sparse.to_dense(best_mask), tf.float32), [-1])

        other_mask = 1. - best_mask
        other_mask = other_mask > 0.
        other_abbox_yx = tf.boolean_mask(abbox_yx, other_mask)
        other_abbox_hw = tf.boolean_mask(abbox_hw, other_mask)
        other_armbbox_yx = tf.boolean_mask(armbbox_yx, other_mask)
        other_armbbox_hw = tf.boolean_mask(armbbox_hw, other_mask)
        other_armconf = tf.boolean_mask(armconf, other_mask)
        other_odmbbox_yx = tf.boolean_mask(odmbbox_yx, other_mask)
        other_odmbbox_hw = tf.boolean_mask(odmbbox_hw, other_mask)
        other_odmconf = tf.boolean_mask(odmconf, other_mask)

        agiou_rate = tf.transpose(gaiou_rate)
        other_agiou_rate = tf.boolean_mask(agiou_rate, other_mask)
        best_agiou_rate = tf.reduce_max(other_agiou_rate, axis=1)
        pos_agiou_mask = best_agiou_rate > 0.5
        neg_agiou_mask = (1. - tf.cast(pos_agiou_mask, tf.float32)) > 0.
        rgindex = tf.argmax(other_agiou_rate, axis=1)

        pos_rgindex = tf.boolean_mask(rgindex, pos_agiou_mask)
        pos_abbox_yx = tf.boolean_mask(other_abbox_yx, pos_agiou_mask)
        pos_abbox_hw = tf.boolean_mask(other_abbox_hw, pos_agiou_mask)
        pos_gbbox_yx = tf.gather(gbbox_yx, pos_rgindex)
        pos_gbbox_hw = tf.gather(gbbox_hw, pos_rgindex)
        pos_armbbox_yx = tf.boolean_mask(other_armbbox_yx, pos_agiou_mask)
        pos_armbbox_hw = tf.boolean_mask(other_armbbox_hw, pos_agiou_mask)
        pos_armconf = tf.boolean_mask(other_armconf, pos_agiou_mask)
        pos_odmbbox_yx = tf.boolean_mask(other_odmbbox_yx, pos_agiou_mask)
        pos_odmbbox_hw = tf.boolean_mask(other_odmbbox_hw, pos_agiou_mask)
        pos_odmconf = tf.boolean_mask(other_odmconf, pos_agiou_mask)
        pos_odmlabel = tf.gather(label, pos_rgindex)
        neg_armconf = tf.boolean_mask(other_armconf, neg_agiou_mask)
        neg_odmconf = tf.boolean_mask(other_odmconf, neg_agiou_mask)

        pos_gbbox_yx = tf.concat([gbbox_yx, pos_gbbox_yx], axis=0)
        pos_gbbox_hw = tf.concat([gbbox_hw, pos_gbbox_hw], axis=0)
        pos_arm_abbox_yx = tf.concat([best_abbox_yx, pos_abbox_yx], axis=0)
        pos_arm_abbox_hw = tf.concat([best_abbox_hw, pos_abbox_hw], axis=0)
        pos_armbbox_yx = tf.concat([best_armbbox_yx, pos_armbbox_yx], axis=0)
        pos_armbbox_hw = tf.concat([best_armbbox_hw, pos_armbbox_hw], axis=0)
        pos_armconf = tf.concat([best_armconf, pos_armconf], axis=0)
        pos_odmbbox_yx = tf.concat([best_odmbbox_yx, pos_odmbbox_yx], axis=0)
        pos_odmbbox_hw = tf.concat([best_odmbbox_hw, pos_odmbbox_hw], axis=0)
        pos_odmlabel = tf.reshape(tf.concat([label, pos_odmlabel], axis=0), [-1])
        pos_odmconf = tf.concat([best_odmconf, pos_odmconf], axis=0)

        pos_shape = tf.shape(pos_armconf)
        neg_armshape = tf.shape(neg_armconf)
        pos_armlabel = tf.constant([0])
        pos_armlabel = tf.tile(pos_armlabel, [pos_shape[0]])
        neg_armlabel = tf.constant([1])
        neg_armlabel = tf.tile(neg_armlabel, [neg_armshape[0]])
        pos_conf_armloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos_armlabel, logits=pos_armconf)
        neg_conf_armloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=neg_armlabel, logits=neg_armconf)
        conf_armloss = tf.reduce_mean(tf.concat([pos_conf_armloss, neg_conf_armloss], axis=-1))

        arm_filter_mask = neg_armconf[:, 1] < 0.99
        neg_odmconf = tf.boolean_mask(neg_odmconf, arm_filter_mask)
        neg_shape = tf.shape(neg_odmconf)
        num_pos = pos_shape[0]
        num_odmneg = neg_shape[0]
        chosen_num_neg = tf.cond(num_odmneg > 3*num_pos, lambda: 3*num_pos, lambda: num_odmneg)
        neg_odmlabel = tf.constant([self.num_classes-1])
        neg_odmlabel = tf.tile(neg_odmlabel, [num_odmneg])

        total_neg_odmloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=neg_odmlabel, logits=neg_odmconf)
        chosen_neg_odmloss, _ = tf.nn.top_k(total_neg_odmloss, chosen_num_neg)

        pos_conf_odmloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos_odmlabel, logits=pos_odmconf)
        conf_odmloss = tf.reduce_mean(tf.concat([pos_conf_odmloss, chosen_neg_odmloss], axis=-1))

        pos_truth_armbbox_yx = (pos_gbbox_yx - pos_arm_abbox_yx) / pos_arm_abbox_hw
        pos_truth_armbbox_hw = tf.log(pos_gbbox_hw / pos_arm_abbox_hw)
        pos_yx_armloss = tf.reduce_sum(self._smooth_l1_loss(pos_armbbox_yx - pos_truth_armbbox_yx), axis=-1)
        pos_hw_armloss = tf.reduce_sum(self._smooth_l1_loss(pos_armbbox_hw - pos_truth_armbbox_hw), axis=-1)
        pos_coord_armloss = tf.reduce_mean(pos_yx_armloss + pos_hw_armloss)

        pos_odm_abbox_yx = pos_arm_abbox_hw * pos_armbbox_yx + pos_arm_abbox_yx
        pos_odm_abbox_hw = tf.exp(pos_armbbox_hw) * pos_arm_abbox_hw
        pos_truth_odmbbox_yx = (pos_gbbox_yx - pos_odm_abbox_yx) / pos_odm_abbox_hw
        pos_truth_odmbbox_hw = tf.log(pos_gbbox_hw / pos_odm_abbox_hw)
        pos_yx_odmloss = tf.reduce_sum(self._smooth_l1_loss(pos_odmbbox_yx - pos_truth_odmbbox_yx), axis=-1)
        pos_hw_odmloss = tf.reduce_sum(self._smooth_l1_loss(pos_odmbbox_hw - pos_truth_odmbbox_hw), axis=-1)
        pos_coord_odmloss = tf.reduce_mean(pos_yx_odmloss + pos_hw_odmloss)

        total_loss = conf_armloss + conf_odmloss + pos_coord_armloss + pos_coord_odmloss
        return total_loss

    def _smooth_l1_loss(self, x):
        return tf.where(tf.abs(x) < 1., 0.5*x*x, tf.abs(x)-0.5)

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'train':
            self.sess.run(self.train_initializer)

    def _create_saver(self):
        weights = tf.trainable_variables(scope='feature_extractor') + tf.trainable_variables('ARM') + tf.trainable_variables('TCB') + tf.trainable_variables('ODM')
        self.saver = tf.train.Saver(weights)
        self.best_saver = tf.train.Saver(weights)

    def _create_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def train_one_epoch(self, lr):
        self.is_training = True
        mean_loss = []
        num_iters = self.num_train // self.batch_size
        for i in range(num_iters):
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.lr: lr})
            sys.stdout.write('\r>> ' + 'iters '+str(i+1)+str('/')+str(num_iters)+' loss '+str(loss))
            sys.stdout.flush()
            mean_loss.append(loss)
        sys.stdout.write('\n')
        mean_loss = np.mean(mean_loss)
        return mean_loss

    def test_one_image(self, images):
        self.is_training = False
        pred = self.sess.run(self.detection_pred, feed_dict={self.images: images})
        return pred

    def save_weight(self, mode, path):
        assert(mode in ['latest', 'best'])
        if mode == 'latest':
            saver = self.saver
        else:
            saver = self.best_saver
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')
        saver.save(self.sess, path, global_step=self.global_step)
        print('save', mode, 'model in', path, 'successfully')

    def load_weight(self, path):
        self.saver.restore(self.sess, path)
        print('load weight', path, 'successfully')

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _load_conv_layer(self, bottom, filters, bias, name):
        if self.data_format == 'channels_last':
            data_format = 'NHWC'
        else:
            data_format = 'NCHW'
        conv = tf.nn.conv2d(bottom, filter=filters, strides=[1, 1, 1, 1], name="kernel"+name, padding="SAME", data_format=data_format)
        conv_bias = tf.nn.bias_add(conv, bias=bias, name="bias"+name)
        return tf.nn.relu(conv_bias)

    def _conv_layer(self, bottom, filters, kernel_size, strides, name=None, dilation_rate=1, activation=None):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
        )
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _dconv_layer(self, bottom, filters, kernel_size, strides, name=None, activation=None):
        conv = tf.layers.conv2d_transpose(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format,
        )
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _max_pooling(self, bottom, pool_size, strides, name=None):
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name=None):
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _dropout(self, bottom, name=None):
        return tf.layers.dropout(
            inputs=bottom,
            rate=self.prob,
            training=self.is_training,
            name=name
        )
