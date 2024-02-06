#!/usr/bin/env python3

"""This module contains the function that
builds, trains, and saves a neural network model in tensorflow
using Adam optimization, mini-batch gradient descent,
learning rate decay, and batch normalization:
"""

import tensorflow as tf
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    builds, trains, and saves a neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent,
    learning rate decay, and batch normalization
    Data_train - tuple with training inputs and training labels, respectively
    Data_valid - tuple with validation inputs and validation labels,
    layers - is a list with the no. of nodes in each layer of the network
    activations - list with the activation functions for each layer
    alpha - is the learning rate
    beta1 - is the weight used for the first moment
    beta2 - is the weight used for the second moment
    epsilon - is a small number to avoid division by zero
    decay_rate - is the decay rate for inverse time decay of the learning rate
    batch_size - is the number of data points in a batch
    epochs - no. of times the training should pass through the whole dataset
    save_path - is the path where the model should be saved to
    Returns: the path where the model was saved
    """

    # Unpack the data
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # Placeholder creation
    x = tf.placeholder(tf.float32, [None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, [None, Y_train.shape[1]], name='y')

    # Initial layer setup
    A = x
    for i, n_units in enumerate(layers):
        with tf.variable_scope(f'layer{i}'):
            weights = tf.get_variable('weights', [A.get_shape().as_list()[1], n_units],
                                      initializer=tf.contrib.layers.variance_scaling_initializer())
            biases = tf.get_variable(
                'biases', [n_units], initializer=tf.constant_initializer(0.0))
            Z = tf.add(tf.matmul(A, weights), biases)

            if i < len(layers) - 1:  # Apply batch normalization to hidden layers
                Z = tf.layers.batch_normalization(Z, training=True)

            A = activations[i](Z)

    # Loss, optimizer, and training operations
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=A, labels=y))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, decay_rate, 1, staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate, beta1, beta2, epsilon).minimize(loss, global_step=global_step)

    # Accuracy calculation
    correct_pred = tf.equal(tf.argmax(A, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Saver and session initialization
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            X_shuffled, Y_shuffled = shuffle_data(
                X_train, Y_train)  # Assuming shuffle_data is defined
            for batch in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[batch:batch + batch_size]
                Y_batch = Y_shuffled[batch:batch + batch_size]
                sess.run(optimizer, feed_dict={x: X_batch, y: Y_batch})

            # Epoch completion log
            train_loss, train_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_loss, valid_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}')

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
