#!/usr/bin/env python3

"""This module contains a function that
builds, trains, and saves a neural network classifier
"""

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    x_train - contains training input data
    y_train - contains training labels
    x_valid - contains validation input data
    y_valid - contains validation labels
    layer_sizes - list with no. of nodes in each layer
    activations - list with activation functions for each layer
    alpha - learning rate
    iterations - no. of iterations to train over
    save_path - designates where to save the model
    graphs collection: - placeholders x & y
                      - tensors(y_pred, loss, accuracy)
                      - train_op
    returns the path where the model was saved
    """

    # create placeholders
    nx = X_train.shape[1]  # Number of features
    classes = Y_train.shape[1]  # Number of classes
    X, Y = create_placeholders(nx, classes)

    # Step 2: Build the Neural Network
    y_pred = forward_prop(X, layer_sizes, activations)

    # Step 3: Calculate accuracy
    accuracy = calculate_accuracy(Y, y_pred)

    # Step 4: Define the Loss Function
    loss = calculate_loss(Y, y_pred)

    # Step 5: Define the Optimizer & train op
    train_op = create_train_op(loss, alpha)

    # Step 5: Initialize Global Variables
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Step 6: Start TensorFlow Session
    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        for epoch in range(iterations + 1):
            epoch_loss = 0
            # Code to process x_train, y_train in batches
            # And run optimizer and calculate loss

            _, epoch_loss = sess.run([train_op, loss], feed_dict={
                                     X: X_train, Y: Y_train})

            epoch_accuracy = sess.run(
                accuracy, feed_dict={X: X_train, Y: Y_train})

            valid_loss = sess.run(loss, feed_dict={X: X_valid, Y: Y_valid})

            valid_accuracy = sess.run(
                accuracy, feed_dict={X: X_valid, Y: Y_valid})

            if epoch % 100 == 0:
                print("After {} iterations:".format(epoch))
                print("    Training Cost: {}".format(epoch_loss))
                print("    Training Accuracy: {}".format(
                    epoch_accuracy))
                print("    Validation Cost: {}".format(valid_loss))
                print("    Validation Accuracy: {}".format(
                    valid_accuracy))

                print("After {} iterations: \n\tTraining Cost: {} \n\tTraining Accuracy: {} \n\tValidation Cost: {} \n\tValidation Accuracy: {}".format(
                    epoch, epoch_loss, epoch_accuracy, valid_loss, valid_accuracy))

        save_path = saver.save(sess, save_path)

    return save_path
