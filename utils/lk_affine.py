#encoding:utf-8

import tensorflow as tf 

def affine(image, transmat):
    # image: tensor
    # transmat: tf.float32, 2x3, pts->avg_pts
    # return: aligned image tensor (h,w:640,512)

    # read image

    img_data = tf.image.convert_image_dtype(image, dtype=tf.float32)
    transmat = tf.cast(transmat, tf.float32)

    in_shape = tf.shape(img_data)
    out_shape = tf.constant([640, 512, 3])
    in_shape = tf.cast(in_shape, tf.float32)
    out_shape = tf.cast(out_shape, tf.float32)

    # adapt to width or heigth
    ratio = tf.div(out_shape, in_shape)
    condition = tf.greater(ratio[0], ratio[1])

    in_shape_new = tf.cond(condition, lambda: (out_shape[1] * in_shape[0] / in_shape[1], out_shape[1], out_shape[2]),
                                      lambda: (out_shape[0], out_shape[0] * in_shape[1] / in_shape[0], out_shape[2]))
    # scale mat
    s = tf.diag(tf.div(in_shape, in_shape_new))
    # calc inv mat
    trans = tf.matmul(transmat, s)
    r = tf.slice(trans, [0, 0], [2, 2])
    t = tf.slice(trans, [0, 2], [2, 1])

    inv_r = tf.matrix_inverse(r)
    inv_t = -tf.matmul(inv_r, t)
    z_f = tf.constant([0.0, 0.0])
    inv_trans = tf.concat([inv_r, inv_t], 1)

    # reshape to a vector of length 8
    transforms = tf.concat([tf.reshape(inv_trans, [6]), z_f], 0 )
    #transforms = [1,0,0,0,1,0,0,0]
    in_shape_new = tf.cast(in_shape_new, tf.int32)
    out_shape = tf.cast(out_shape, tf.int32)

    # calculate padding
    z = tf.constant([0, 0])
    p = tf.subtract(out_shape, in_shape_new)[:2]
    r_p = tf.reverse(p, [0])
    paddings = tf.cond(condition, lambda: tf.stack([r_p, z, z]), lambda: tf.stack([z, p, z]))

    # resize, pad, transform
    resize_img = tf.image.resize_images(img_data, in_shape_new[:2], method=0)
    pad_img = tf.pad(resize_img, paddings, "CONSTANT")
    transformed_img = tf.contrib.image.transform(pad_img, transforms)

    return transformed_img