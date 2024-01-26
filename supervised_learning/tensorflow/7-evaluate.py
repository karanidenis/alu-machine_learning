#!/usr/bin/env python3

"""This module contains a function that
evaluates the output of a neural network
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    x - contains input data to evaluate
    y -  contains one-hot labels for x
    save_path - location to load the model from
    returns the network's prediction, accuracy, and loss
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('X')[0]
        y = tf.get_collection('Y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
    return (prediction, accuracy, loss)
