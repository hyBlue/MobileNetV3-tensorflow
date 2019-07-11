import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np

# Weight initializers
he = variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
random_normal = tf.initializers.random_normal


def conv_bn(inp, oup, stride, conv=tf.layers.conv2d, norm=tf.layers.batch_normalization, nlin=tf.nn.relu):
    with tf.variable_scope('conv_bn'):
        conv_layer = conv(inp, oup, 3, (stride, stride), padding='same', use_bias=False, kernel_initializer=he)
        norm_layer = norm(conv_layer)
        nlin_layer = nlin(norm_layer)
        return nlin_layer


def conv_1x1_bn(inp, oup, conv=tf.layers.conv2d, norm=tf.layers.batch_normalization, nlin=tf.nn.relu):
    with tf.variable_scope('conv_1x1_bn'):
        conv_layer = conv(inp, oup, 1, (1, 1), padding='valid', use_bias=False, kernel_initializer=he)
        norm_layer = norm(conv_layer)
        nlin_layer = nlin(norm_layer)
        return nlin_layer


def Hswish(inp):
    return inp * tf.nn.relu6(inp + 3.) / 6


def Hsigmoid(inp):
    return tf.nn.relu6(inp + 3.) / 6


class SEModule:

    def __init__(self, channel, reduction=4):
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.lin = tf.layers.dense
        self.channel = channel
        self.reduction = reduction

    def __call__(self, inp):
        b, _, _, c = inp.shape
        y = self.avg_pool(inp)
        y = tf.reshape(y, (b, c))

        y = tf.layers.dense(y, self.channel // self.reduction, activation=tf.nn.relu, use_bias=False,
                            kernel_initializer=random_normal)
        y = tf.layers.dense(y, self.channel, activation=Hsigmoid, use_bias=False, kernel_initializer=random_normal)
        y = tf.reshape(y, (b, 1, 1, c))
        return inp * tf.broadcast_to(y, inp.shape)


def identity(inp):
    return inp


class MobileBottleneck:

    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        self.use_res_connect = stride == 1 and inp == oup
        self.inp = inp
        if nl == 'RE':
            nlin_layer = tf.nn.relu
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError

        if se:
            SELayer = SEModule(exp)
        else:
            SELayer = identity

        with tf.variable_scope('MobileBottleneck'):
            x = tf.layers.conv2d(inp, exp, 1, 1, 'valid', use_bias=False, kernel_initializer=he)
            x = tf.layers.batch_normalization(x)
            x = nlin_layer(x)
            x = tf.layers.separable_conv2d(x, exp, kernel, stride, 'same', use_bias=False, depthwise_initializer=he)
            x = tf.layers.batch_normalization(x)
            x = SELayer(x)
            x = nlin_layer(x)
            x = tf.layers.conv2d(x, oup, 1, 1, 'valid', use_bias=False, kernel_initializer=he)
            x = tf.layers.batch_normalization(x)
        self.conv = x

    def __call__(self):

        if self.use_res_connect:
            return self.inp + self.conv
        else:
            return self.conv


def make_divisible(inp, divisible_by=8):
    return int(np.ceil(inp * 1. / divisible_by) * divisible_by)


class MobileNetV3:

    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'RE', 2],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
        else:
            raise NotImplementedError
        self.n_class = n_class
        self.mode = mode
        self.width_mult = width_mult
        self.mobile_setting = mobile_setting
        self.input_channel = 16
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.p = dropout

    def __call__(self, inp):
        x = conv_bn(inp, self.input_channel, 2, nlin=Hswish)

        for i, (k, exp, c, se, nl, s) in enumerate(self.mobile_setting):
            with tf.variable_scope(f'BottleneckLayer_{i}', reuse=False):
                output_channel = make_divisible(c * self.width_mult)
                exp_channel = make_divisible(exp * self.width_mult)
                x = MobileBottleneck(x, output_channel, k, s, exp_channel, se, nl)()

        if self.mode == 'large':
            last_conv = make_divisible(960 * self.width_mult)
            x = conv_1x1_bn(x, last_conv, nlin=Hswish)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.layers.conv2d(x, self.last_channel, 1, 1, 'valid', kernel_initializer=he)
            x = Hswish(x)

        elif self.mode == 'small':
            last_conv = make_divisible(576 * self.width_mult)

            x = conv_1x1_bn(x, last_conv, nlin=Hswish)
            b, _, _, c = x.shape
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.reshape(x, (b, 1, 1, c))
            x = tf.layers.conv2d(x, self.last_channel, 1, 1, 'valid')

        with tf.variable_scope('classifier'):
            x = tf.layers.dropout(x, rate=self.p)
            x = tf.reduce_mean(x, axis=[1, 2])
            x = tf.layers.dense(x, self.n_class, kernel_initializer=random_normal)

        return x


if __name__ == '__main__':
    inp = tf.get_variable('input', [1, 224, 224, 3], dtype=tf.float32)
    print(MobileNetV3()(inp))

