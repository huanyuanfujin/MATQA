import tensorflow as tf
import numpy as np
#

BN_EPSILON = 0.001


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def batch_normalization_layer(input_layer, dimension):

    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    # out_channel = filter_shape[-1]
    out_channel = filter_shape[1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def inference(input_tensor_batch, n, reuse):
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        # conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 300, 160], 1)  # CDE
        conv0 = conv_bn_relu_layer(input_tensor_batch, [1, 1, 1, 1], 1)  # NEWs
        print('conv0:', conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' % i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 1, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 1)
            layers.append(conv1)
    print('conv1:', layers[-1])

    for i in range(n):
        with tf.variable_scope('conv2_%d' % i, reuse=reuse):
            conv2 = residual_block(layers[-1], 1)  # NEWs
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' % i, reuse=reuse):
            conv3 = residual_block(layers[-1], 1)
            layers.append(conv3)
        print('shape of conv3:', conv3.get_shape())
        # assert conv3.get_shape().as_list()[1:] == [8, 8, 64]
    # last_year = layers[-1]
    # last_year = tf.reshape(last_year, (last_year.shape[0], last_year.shape[1], last_year.shape[2]))
    # print('last_year:', last_year.get_shape())
    # fc1_w = tf.get_variable('w', [300, 60], initializer=tf.truncated_normal_initializer(stddev=0.1))
    # fc1_b = tf.get_variable('b', [60], initializer=tf.constant_initializer(0.1))
    # last_year_feature = tf.matmul(last_year, fc1_w) + fc1_b
    # print('last_year_feature:', last_year_feature.get_shape())
    return layers[-1]


def tet_graph():
    input_tensor = tf.constant(np.ones([32, 96, 128, 1]), dtype=tf.float32)
    result = inference(input_tensor, 1, reuse=False)
    print('yes:', result)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


if __name__ == '__main__':
    tet_graph()