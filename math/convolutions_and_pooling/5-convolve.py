#!/usr/bin/env python3

"""
This module has the method that performs
convolution on grayscale images
with multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Convolution with multiple kernels
    c - no. of channels
    nc - no. of kernels/ filters
    """
    # m, h, w, c = images.shape
    # kh, kw, _, nc = kernels.shape
    # # stride
    # sh, sw = stride

    # if padding == 'same':
    #     ph = max(0, (h - 1) * stride[0] + kh - h)
    #     pw = max(0, (h-1) * stride[1] + kw - w)
    # elif padding == 'valid':
    #     ph, pw = 0, 0
    # elif isinstance(padding, tuple):
    #     ph, pw = padding

    # # Calculate the output dimensions
    # oh = int((h - kh + 2 * ph) / sh) + 1
    # ow = int((w - kw + 2 * pw) / sw) + 1

    # # Pad the input images
    # images_padded = np.pad(
    #     images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # # Initialize the output tensor
    # convolved_images = np.zeros((m, oh, ow, nc))

    # # Perform the convolution
    # for i in range(oh):
    #     for j in range(ow):
    #         for k in range(nc):
    #             # Extract a patch from the image
    #             patch = images_padded[:, i * sh:i *
    #                                   sh + kh, j * sw:j * sw + kw, :]
    #             # Convolve using tensordot to handle multiple channels
    #             convolved_images[:, i, j, k] = np.tensordot(
    #                 patch, kernels[:, :, :, k], axes=([1, 2, 3], [0, 1, 2]))

    # return convolved_images

    kh, kw, kc, nc = kernels.shape
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
    convoluted = np.zeros((m, ch, cw, nc))
    for c in range(nc):
        for h in range(ch):
            for w in range(cw):
                square = padded[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :]
                insert = np.sum(square * kernels[..., c], axis=(1, 2, 3))
                convoluted[:, h, w, c] = insert
    return convoluted
