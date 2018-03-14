import tensorflow as tf

from config import cfg
from capslayer.utils import softmax
from LoadData import get_batch_data
import capslayer.layers


class CapsNet(object):
    def __init__(self, height=28, width=28, channels=1, num_label=10):
        """
        Args:
            height: Integer, the height of input.
            width: Integer, the width of input.
            channels: Integer, the channels of input.
            num_label: Integer, the category number.
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

        self.graph = tf.Graph()
        with self.graph.as_default():
            if cfg.is_training:
                self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size)
                self.x = tf.reshape(self.X, shape=[cfg.batch_size, self.height, self.width, self.channels])
                self.Y = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()

                self.global_step = tf.Variable(1, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            else:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, None))
                self.x = tf.reshape(self.X, shape=[cfg.batch_size, self.height, self.width, self.channels])
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
                self.Y = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)
                self.build_arch()

            with tf.variable_scope('accuracy'):
                logits_idx = tf.to_int32(tf.argmax(softmax(self.activation, axis=1), axis=1))
                correct_prediction = tf.equal(tf.to_int32(self.labels), logits_idx)
                self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])

            if cfg.is_training:
                self._summary()

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.contrib.layers.conv2d(self.x, num_outputs=64, kernel_size=3, stride=1, padding='VALID')

        with tf.variable_scope('Conv2_layer'):
            conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=64, kernel_size=3, stride=1, padding='VALID')

        with tf.variable_scope('Pool_layer'):
            pool = tf.contrib.layers.avg_pool2d(conv2, kernel_size=2, stride=1)

        with tf.variable_scope('Conv3_layer'):
            conv3 = tf.contrib.layers.conv2d(pool, num_outputs=256, kernel_size=3, stride=1, padding='VALID')

        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps, activation = capslayer.layers.primaryCaps(conv3, filters=32, kernel_size=3, strides=1,
                                                                   out_caps_shape=[8, 1])

        with tf.variable_scope('DigitCaps_layer'):
            primaryCaps = tf.reshape(primaryCaps, shape=[cfg.batch_size, -1, 8, 1])
            self.digitCaps, self.activation = capslayer.layers.fully_connected(primaryCaps, activation, num_outputs=10,
                                                                               out_caps_shape=[16, 1],
                                                                               routing_method='DynamicRouting')

    def loss(self):

        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.activation))
        max_r = tf.square(tf.maximum(0., self.activation - cfg.m_minus))

        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        T_c = self.Y
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

    def _summary(self):
        train_summary = [tf.summary.scalar('train/margin_loss', self.margin_loss),
                         tf.summary.scalar('train/accuracy', self.test_acc),
                         tf.summary.histogram('activation', self.activation)]
        self.train_summary = tf.summary.merge(train_summary)
