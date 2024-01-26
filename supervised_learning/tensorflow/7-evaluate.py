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

    export_dir = save_path

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.Session(graph=tf.Graph()) as sess:
        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.TRAINING],
                                             signature_def_map=foo_signatures,
                                             assets_collection=foo_assets,
                                             strip_default_attrs=True)

    # Add a second MetaGraphDef for inference.
    with tf.Session(graph=tf.Graph()) as sess:
        builder.add_meta_graph([tag_constants.SERVING],
                               strip_default_attrs=True)

    builder.save()
