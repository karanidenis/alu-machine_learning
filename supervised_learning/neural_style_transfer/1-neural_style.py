#!/usr/bin/env python3

"""
This module contains a class NST with functions that
creates a model used to calculate cost
"""
import numpy as np
import tensorflow as tf


class NST:
    """class NST"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """class constructor
        style_image - image used as style reference stored as numpy array
        content_image - image used as content reference stored as numpy array
        alpha - weight for content cost
        beta - weight for style cost
        model - the Keras model used to calculate cost
        """
        # tensorflow executes eagerly
        tf.enable_eager_execution()

        error = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(error)
        if style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(error)

        error = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(content_image, np.ndarray):
            raise TypeError(error)
        if content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(error)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        # define a model
        model = tf.keras.applications.vgg19.VGG19(include_top=False)

    @staticmethod
    def scale_image(image):
        """rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        image - numpy.ndarray of shape (h, w, 3) with image to be scaled
        Returns a scaled image with shape (h, w, 3) - max(h, w) is 512 pixels
        """
        err = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray):
            raise TypeError(err)
        if image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(err)

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * h_new / h)
        else:
            w_new = 512
            h_new = int(h * w_new / w)

        image = image[tf.newaxis, ...]
        image = tf.image.resize_bicubic(image, (h_new, w_new))
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        return image

    def load_model(self):
        """creates the model used to calculate cost
        model should output the style and content layers"""
        base_model = tf.keras.applications.vgg19.VGG19(include_top=False)
        base_model.trainable = False
        style_outputs = [base_model.get_layer(name).output
                         for name in self.style_layers]
        content_outputs = base_model.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_outputs]
        model = tf.keras.models.Model(base_model.input, model_outputs)
        return model