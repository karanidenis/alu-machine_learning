#!/usr/bin/env python3

"""
This module has the method that performs
convolution on grayscale images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    convolution with channels
    """
    # m, h, w, c = images.shape
    # kh, kw, kc = kernel.shape
    # # stride
    # sh, sw = stride

    # if padding == 'same':
    #     ph = int(((h - 1) * sh + kh - h) / 2)
    #     pw = int(((w - 1) * sw + kw - w) / 2)
    # elif isinstance(padding, tuple):
    #     ph, pw = padding
    # elif padding == 'valid':
    #     ph, pw = 0, 0

    # # output
    # out_h = int((h - kh + 2 * ph) / sh) + 1
    # out_w = int((w - kw + 2 * pw) / sw) + 1

    # # Pad the input images
    # images_padded = np.pad(
    #     images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # # initialise convolved images
    # convolved_images = np.zeros((m, out_h, out_w))

    # # Perform the convolution
    # for i in range(out_h):
    #     for j in range(out_w):
    #         # Extract a patch from the image
    #       patch = images_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
    #         convolved_images[:, i, j] = np.tensordot(
    #             patch, kernel, axes=([1, 2, 3], [0, 1, 2]))

    # return convolved_images

    kh, kw, kc = kernel.shape
    m, hm, wm, cm = images.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((hm - 1) * sh + kh - hm) / 2) + 1
        pw = int(((wm - 1) * sw + kw - wm) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    ch = int((hm + 2 * ph - kh) / sh) + 1
    cw = int((wm + 2 * pw - kw) / sw) + 1
    convoluted = np.zeros((m, ch, cw))
    for h in range(ch):
        for w in range(cw):
            square = padded[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :]
            insert = np.sum(square * kernel, axis=1).sum(axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted
