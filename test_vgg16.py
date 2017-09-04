import numpy as np
import tensorflow as tf

import vgg16
import utils

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

vgg = vgg16.Vgg16()
images = tf.placeholder("float", [2, 224, 224, 3])
vgg.build(images)
feature_map = sess.run(vgg.pool5,feed_dict={images:batch})
print(feature_map)
print(tf.size(feature_map))
'''
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    print(2)
    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    print(3)
    print(prob)
    utils.print_prob(prob[0], './synset.txt')
    utils.print_prob(prob[1], './synset.txt')
'''