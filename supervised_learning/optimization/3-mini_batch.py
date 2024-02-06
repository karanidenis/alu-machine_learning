#!/usr/bin/env python3

"""This module contains the function that
that trains a loaded neural network model using
mini-batch gradient descent:
"""

import tensorflow as tf
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model using mini-batch gradient descent
    X_train - is a numpy.ndarray of shape (m, 784) containing the training data
        m - is the number of data points
        784 - is the number of input features
    Y_train - is a one-hot numpy.ndarray of shape (m, 10) containing the training labels
        10 - is the number of classes the model should classify
    X_valid - is a numpy.ndarray of shape (m, 784) containing the validation data
    Y_valid - is a one-hot numpy.ndarray of shape (m, 10) containing the validation labels
    batch_size - is the number of data points in a batch
    epochs - is the number of times the training should pass through the whole dataset
    load_path - is the path from which to load the model
    save_path - is the path to where the model should be saved after training
    Returns: the path where the model was saved"""

    # shuffle the data
    X_train, Y_train = shuffle_data(X_train, Y_train)

    m = X_train.shape[0]
    # create placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y')

    # mini-batch
    mini_batches = []
    for i in range(0, m, batch_size):
        X_mini = X_train[i:i + batch_size]
        Y_mini = Y_train[i:i + batch_size]
        mini_batches.append((X_mini, Y_mini))

    # load the model
    model = tf.train.import_meta_graph(load_path + '.meta')

    # create a saver
    saver = tf.train.Saver()

    # create a session
    with tf.Session() as sess:
        # restore the model
        model.restore(sess, load_path)

        # get the tensors by their variable name
        x = tf.get_default_graph().get_tensor_by_name("x:0")
        y = tf.get_default_graph().get_tensor_by_name("y:0")
        accuracy = tf.get_default_graph().get_tensor_by_name("Mean_1:0")
        loss = tf.get_default_graph().get_tensor_by_name("Mean:0")
        train_op = tf.get_default_graph().get_operation_by_name("train_op")

        # train the model
        for epoch in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_train = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            cost_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(acc_train))
            print("\tValidation Cost: {}".format(cost_valid))
            print("\tValidation Accuracy: {}".format(acc_valid))
            if epoch < epochs:
                for X_mini, Y_mini in mini_batches:
                    sess.run(train_op, feed_dict={x: X_mini, y: Y_mini})
                    cost_train = sess.run(
                        loss, feed_dict={x: X_train, y: Y_train})
                    acc_train = sess.run(accuracy, feed_dict={
                                         x: X_train, y: Y_train})
                    cost_valid = sess.run(
                        loss, feed_dict={x: X_valid, y: Y_valid})
                    acc_valid = sess.run(accuracy, feed_dict={
                                         x: X_valid, y: Y_valid})
                    print("\tTraining Cost: {}".format(cost_train))
                    print("\tTraining Accuracy: {}".format(acc_train))
                    print("\tValidation Cost: {}".format(cost_valid))
                    print("\tValidation Accuracy: {}".format(acc_valid))
        # save the model
        save_path = saver.save(sess, save_path)

    return save_path
