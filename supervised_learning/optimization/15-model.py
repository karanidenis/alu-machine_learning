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

    # Create placeholders
    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='x')
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name='y')

    # Create the neural network
    A = create_layer(x, layers[0], activations[0])
    for i in range(1, len(layers)):
        A = create_batch_norm_layer(A, layers[i], activations[i])

    # Create the loss function
    loss = tf.losses.softmax_cross_entropy(y, A)

    # Create the learning rate decay operation
    global_step = tf.Variable(0, trainable=False)
    alpha = tf.train.inverse_time_decay(alpha,
            global_step, decay_rate, 1, staircase=True)

    # Create the Adam optimization operation
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Create the accuracy calculation operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(A, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float 32))
    # Create the saver
    saver = tf.train.Saver()

    # Create the session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Calculate the number of batches for training and validation
    m = X_train.shape[0]
    num_batches = m // batch_size if m % batch_size == 0 else (m // batch_size) + 1
    m_valid = X_valid.shape[0]
    num_batches_valid = m_valid // batch_size if m_valid % batch_size == 0 else (m_valid // batch_size) + 1

    # Training loop
    for epoch in range(epochs):
        # Shuffle data at the start of each epoch
        X_train, Y_train = shuffle_data(X_train, Y_train)
        for i in range(num_batches):
            start_i = i * batch_size
            end_i = start_i + batch_size
            X_mini = X_train[start_i:end_i]
            Y_mini = Y_train[start_i:end_i]
            session.run(train_op, feed_dict={x: X_mini, y: Y_mini})
            if i % 100 == 0 and i != 0:
                loss_train = session.run(loss,
                                feed_dict={x: X_mini, y: Y_mini})
                accuracy_train = session.run(accuracy, 
                                    feed_dict={x: X_mini, y: Y_mini})
                print("After {} batches: ".format(i))
                print("\tTraining Cost: {}".format(loss_train))
                print("\tTraining Accuracy: {}".format(accuracy_train))
        loss_valid = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
        accuracy_valid = session.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
        print("After {} epochs: ".format(epoch))
        print("\tTraining Cost: {}".format(loss_train))
        print("\tTraining Accuracy: {}".format(accuracy_train))
        print("\tValidation Cost: {}".format(loss_valid))
        print("\tValidation Accuracy: {}".format(accuracy_valid))

    # Save the model
    saved_path = saver.save(session, save_path)
    session.close()

    return saved_path
