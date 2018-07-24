"""
face_net created by sunyi
"""
import tensorflow as tf


# define batch normalization layer
def batch_norm(input_op, is_training, name):
    """
        Batch normalization on convolutional maps.
        Args:
            input_op:           Tensor, 4D NHWC input maps
            is_training: boolean tf.Varialbe, true indicates training phase
            name:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """

    axis = list(range(len(input_op.get_shape()) - 1))
    size = input_op.get_shape()[-1].value

    beta = tf.get_variable(name + "beta",
                             shape=[size], dtype=tf.float32,
                             initializer=tf.zeros_initializer,
                             regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))

    scale = tf.get_variable(name + "scale",
                             shape=[size], dtype=tf.float32,
                             initializer=tf.ones_initializer,
                             regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))

    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(input_op, axis, name='moments', keep_dims=True)
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(tf.cast(is_training, tf.bool),
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(input_op, mean, var, beta, scale, epsilon)

    return normed



# define convolution layer
def conv_op(input_op, name, kh, kw, n_out, dh, dw, is_training, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
            shape=[kh, kw, n_in, n_out], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1),
            padding='SAME')
        biases = tf.get_variable(scope + "b",
                         shape=[n_out], dtype=tf.float32,
                         initializer=tf.zeros_initializer,
                         regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))

        z= tf.nn.bias_add(conv, biases)
        bn = batch_norm(z, is_training=is_training, name=scope)
        activation = tf.nn.relu(bn, name=scope)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('activations', activation)
        p += [kernel, biases]
        return activation

# define fully connectedly layer
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope+"w",
                                  shape=[n_in, n_out], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))

        biases = tf.get_variable(scope+"b",
                               shape=[n_out],dtype=tf.float32,
                               initializer=tf.zeros_initializer,
                               regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))

        fc = tf.add(tf.matmul(input_op, weights), biases, name=scope)
        # activation = tf.nn.relu(fc, name=scope)
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        p += [weights, biases]
        return fc

# define max pooling layer
def mpool_op(input_op, name, kh, kw, dh, dw):

    return tf.nn.max_pool(input_op,
        ksize=[1, kh, kw, 1],
        strides=[1, dh, dw, 1],
        padding='SAME',
        name=name)

# define networks:train net, created by Yi Sun
def face_net_train(input_op,NUM_CLASSES, NUM_FEATURE,is_training):

    # trainable parameter list
    # trainable parameter list
    p = []
    
    # phase 1
    net = conv_op(input_op, name="conv1", kh=3, kw=3, n_out=64, dh=1, dw=1, is_training=is_training, p=p)
    net = mpool_op(net, name="pool1", kh=2, kw=2, dw=2,  dh=2)

    #phase 2
    net = conv_op(net, name="conv2_1", kh=3, kw=3, n_out=96, dh=1, dw=1, is_training=is_training, p=p)
    net = conv_op(net, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, is_training=is_training, p=p)
    net = mpool_op(net, name="pool2", kh=2, kw=2, dw=2,  dh=2)

    # phase 3
    net = conv_op(net, name="conv3_1", kh=3, kw=3, n_out=128, dh=1, dw=1, is_training=is_training, p=p)
    net = conv_op(net, name="conv3_2", kh=3, kw=3, n_out=160, dh=1, dw=1, is_training=is_training, p=p)
    net = conv_op(net, name="conv3_3", kh=3, kw=3, n_out=192, dh=1, dw=1, is_training=is_training, p=p)
    net = mpool_op(net, name="pool3", kh=2, kw=2, dw=2,  dh=2)

    # phase 4
    net = conv_op(net, name="conv4_1", kh=3, kw=3, n_out=192, dh=1, dw=1, is_training=is_training, p=p)
    net = conv_op(net, name="conv4_2", kh=3, kw=3, n_out=224, dh=1, dw=1, is_training=is_training, p=p)
    net = conv_op(net, name="conv4_3", kh=3, kw=3, n_out=256, dh=1, dw=1, is_training=is_training, p=p)
    net = mpool_op(net, name="pool4", kh=2, kw=2, dw=2, dh=2)

    # phase 5
    net = conv_op(net, name="conv5_1", kh=3, kw=3, n_out=256, dh=1, dw=1, is_training=is_training, p=p)
    net = conv_op(net, name="conv5_2", kh=3, kw=3, n_out=256, dh=1, dw=1, is_training=is_training, p=p)
    net = mpool_op(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # phase 6
    net = conv_op(net, name="conv6", kh=3, kw=3, n_out=256, dh=1, dw=1, is_training=is_training, p=p)

    # phase 7: flatten pool4
    shp = net.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    net = tf.reshape(net, [-1, flattened_shape], name="reshp1")
    # test if not drop if the loss will desc ..
    net = tf.cond(is_training,lambda:tf.nn.dropout(net, 0.7, name="conv6_drop"), lambda:net)
    # phase 7: feature layer
    net = fc_op(net, n_out=NUM_FEATURE, name="fc7", p=p)
    
    tf.add_to_collection("output", net)
    net = tf.cond(is_training,lambda:tf.nn.dropout(net, 0.5, name="fc7_drop"), lambda:net)
    # phase 8: network output
    net = fc_op(net, n_out=NUM_CLASSES, name="fc8", p=p)
    tf.add_to_collection("output",net)

    logits = tf.get_collection("output")[1]
    feature = tf.get_collection("output")[0]
    
    return logits, feature

# feature extraction
def face_net_inference(input_op, NUM_CLASSES=14294, NUM_FEATURE=256):

    # trainable parameter list
    p = []
    # phase 1
    conv1 = conv_op(input_op, name="conv1", kh=3, kw=3, n_out=64, dh=1, dw=1, is_training=False, p=p)
    pool1 = mpool_op(conv1, name="pool1", kh=2, kw=2, dw=2,  dh=2)

    #phase 2
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=96, dh=1, dw=1, is_training=False, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, is_training=False, p=p)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dw=2,  dh=2)

    # phase 3
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=128, dh=1, dw=1, is_training=False, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=160, dh=1, dw=1, is_training=False, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=192, dh=1, dw=1, is_training=False, p=p)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dw=2,  dh=2)

    # phase 4
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=192, dh=1, dw=1, is_training=False, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=224, dh=1, dw=1, is_training=False, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=256, dh=1, dw=1, is_training=False, p=p)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dw=2, dh=2)

    # phase 5
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=256, dh=1, dw=1, is_training=False, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=256, dh=1, dw=1, is_training=False, p=p)
    pool5 = mpool_op(conv5_2, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # phase 6
    conv6 = conv_op(pool5, name="conv6", kh=3, kw=3, n_out=256, dh=1, dw=1, is_training=False, p=p)

    # phase 7: flatten pool4
    shp = conv6.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    reshp1 = tf.reshape(conv6, [-1, flattened_shape], name="reshp1")

    # phase 7: feature layer
    fc7 = fc_op(reshp1, n_out=NUM_FEATURE, name="fc7", p=p)

    # phase 8: network output
    fc8 = fc_op(fc7, n_out=NUM_CLASSES, name="fc8", p=p)
    softmax = tf.nn.softmax(fc8)
    feature = fc7
    predictions = tf.argmax(softmax, 1)
    return fc7
